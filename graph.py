"""
graph.py — LangGraph stateful workflow definition.

Graph nodes:
  route   → decide which tool to use (dynamic router)
  execute → run the selected tool and retrieve context
  correct → evaluate confidence; retry + reroute if low
  answer  → synthesize polished final answer with citations

State flows: route → execute → correct → answer
"""

from __future__ import annotations
from typing import TypedDict, Any, Optional
import re
import time

from langgraph.graph import StateGraph, END
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama

from agent.router import route_query
from agent.tools import build_tools
from agent.corrector import self_correct, is_low_confidence, confidence_score


# ── Agent state schema ────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str              # Original user question
    tool_name: str          # Tool selected by router (may change after reroute)
    raw_context: str        # Content retrieved by the tool
    answer: str             # Final synthesized answer
    citations: list[str]    # Source page references extracted from context
    confidence: float       # Confidence score of the final answer (0–1)
    attempts: int           # Total attempts including retries
    elapsed_ms: float       # End-to-end wall-clock time in ms
    error: Optional[str]    # Any error message


# ── Node implementations ──────────────────────────────────────────────────────

def make_route_node(llm):
    def route_node(state: AgentState) -> AgentState:
        tool_name = route_query(state["query"], llm=llm)
        print(f"[Router]    → Selected tool: '{tool_name}'")
        return {**state, "tool_name": tool_name}
    return route_node


def make_execute_node(tools: dict):
    """Initial retrieval — runs the routed tool once to populate raw_context."""
    def execute_node(state: AgentState) -> AgentState:
        tool_fn = tools.get(state["tool_name"], tools["retriever"])
        try:
            raw_context = tool_fn(state["query"])
        except Exception as exc:
            raw_context = f"Tool execution error: {exc}"
        return {**state, "raw_context": raw_context}
    return execute_node


def make_correct_node(tools: dict, llm):
    """
    Synthesise an answer from the retrieved context.
    If confidence is low, trigger retry + reroute loop (corrector.py).
    """
    def synthesize(query: str, context: str) -> str:
        prompt = (
            "You are an expert research assistant. Answer the question below "
            "using ONLY the provided context. Be concise, precise, and cite "
            "page numbers where available.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )
        return str(llm.complete(prompt)).strip()

    def correct_node(state: AgentState) -> AgentState:
        # First attempt uses context already retrieved in execute_node
        initial_answer = synthesize(state["query"], state["raw_context"])
        score = confidence_score(initial_answer)

        if not is_low_confidence(initial_answer):
            # High-confidence on first try — skip retry loop
            return {
                **state,
                "answer": initial_answer,
                "confidence": round(score, 2),
                "attempts": 1,
            }

        # Low confidence → retry + reroute via corrector
        tool_fn = tools.get(state["tool_name"], tools["retriever"])
        final_answer, attempts, final_tool = self_correct(
            query=state["query"],
            tool_fn=tool_fn,
            synthesize_fn=synthesize,
            llm=llm,
            tools=tools,
            original_tool=state["tool_name"],
            max_retries=2,
        )
        return {
            **state,
            "answer": final_answer,
            "tool_name": final_tool,          # reflects actual tool that gave best answer
            "confidence": round(confidence_score(final_answer), 2),
            "attempts": attempts,
        }

    return correct_node


def make_answer_node(llm):
    """Polish the answer and extract structured citations from raw_context."""
    def answer_node(state: AgentState) -> AgentState:
        # Extract page citations from raw_context markers like [Page 3 | text]
        raw = state.get("raw_context", "")
        page_nums = re.findall(r"\[Page (\d+)", raw)
        citations = [f"Page {p}" for p in sorted(set(page_nums), key=int)]

        # Polish answer for clarity
        polish_prompt = (
            "Rewrite the research answer below to be clear, well-structured, "
            "and professional. Preserve all factual content and page references. "
            "Use bullet points for multi-part answers.\n\n"
            f"ANSWER:\n{state['answer']}\n\n"
            "POLISHED ANSWER:"
        )
        try:
            polished = str(llm.complete(polish_prompt)).strip()
            if len(polished) < 20:          # guard against empty LLM output
                polished = state["answer"]
        except Exception:
            polished = state["answer"]

        return {**state, "answer": polished, "citations": citations}

    return answer_node


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(index: VectorStoreIndex, model: str = "llama3") -> Any:
    """
    Build and compile the LangGraph research agent.

    Pipeline:  route → execute → correct → answer → END
    """
    llm = Ollama(model=model, request_timeout=120.0)
    tools = build_tools(index, llm)

    builder = StateGraph(AgentState)

    builder.add_node("route",   make_route_node(llm))
    builder.add_node("execute", make_execute_node(tools))
    builder.add_node("correct", make_correct_node(tools, llm))
    builder.add_node("answer",  make_answer_node(llm))

    builder.set_entry_point("route")
    builder.add_edge("route",   "execute")
    builder.add_edge("execute", "correct")
    builder.add_edge("correct", "answer")
    builder.add_edge("answer",  END)

    return builder.compile()


# ── High-level runner ─────────────────────────────────────────────────────────

def run_query(graph, query: str) -> dict:
    """Run a query through the compiled graph and return the final state dict."""
    t0 = time.time()

    initial_state: AgentState = {
        "query":      query,
        "tool_name":  "",
        "raw_context": "",
        "answer":     "",
        "citations":  [],
        "confidence": 0.0,
        "attempts":   0,
        "elapsed_ms": 0.0,
        "error":      None,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        final_state = {
            **initial_state,
            "error":  str(exc),
            "answer": f"Agent error: {exc}",
        }

    final_state["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    print(
        f"[Agent]     ✅ Done in {final_state['elapsed_ms']}ms | "
        f"tool='{final_state['tool_name']}' | "
        f"attempts={final_state['attempts']} | "
        f"confidence={final_state['confidence']}"
    )
    return final_state
