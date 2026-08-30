"""
tools.py — Tool definitions available to the LangGraph agent.
Each tool is a callable that takes a query string and returns a result string.
"""

from __future__ import annotations
from typing import Any, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore


# ── Retriever tool ────────────────────────────────────────────────────────────

class RetrieverTool:
    """Semantic search over the indexed document."""

    name = "retriever"
    description = (
        "Search the indexed PDF document for relevant text, tables, or figures. "
        "Use this for factual questions, definitions, or when you need raw content."
    )

    def __init__(self, index: VectorStoreIndex, top_k: int = 5):
        self.retriever = index.as_retriever(similarity_top_k=top_k)

    def run(self, query: str) -> str:
        nodes: list[NodeWithScore] = self.retriever.retrieve(query)
        if not nodes:
            return "No relevant content found in the document."

        results = []
        for node in nodes:
            meta = node.metadata
            source_tag = f"[Page {meta.get('page', '?')} | {meta.get('content_type', 'text')}]"
            results.append(f"{source_tag}\n{node.get_content()}")

        return "\n\n---\n\n".join(results)

    def __call__(self, query: str) -> str:
        return self.run(query)


# ── Summarizer tool ───────────────────────────────────────────────────────────

class SummarizerTool:
    """Summarise retrieved content using the LLM."""

    name = "summarizer"
    description = (
        "Summarize a section of the document. Provide the text you want summarized. "
        "Best for long passages or when you need a concise overview."
    )

    def __init__(self, llm):
        self.llm = llm

    def run(self, text: str) -> str:
        prompt = (
            "You are a research assistant. Summarize the following content concisely "
            "while preserving all key facts, figures, and conclusions.\n\n"
            f"CONTENT:\n{text}\n\nSUMMARY:"
        )
        response = self.llm.complete(prompt)
        return str(response)

    def __call__(self, text: str) -> str:
        return self.run(text)


# ── Table analyst tool ────────────────────────────────────────────────────────

class TableAnalystTool:
    """Answer questions that specifically involve tables."""

    name = "table_analyst"
    description = (
        "Analyze data in tables extracted from the PDF. "
        "Use when the question involves numbers, comparisons, or structured data."
    )

    def __init__(self, index: VectorStoreIndex, llm):
        self.index = index
        self.llm = llm
        # Filter retriever to table-type nodes
        self.retriever = index.as_retriever(
            similarity_top_k=5,
            filters={"content_type": "table"},
        )

    def run(self, query: str) -> str:
        try:
            nodes: list[NodeWithScore] = self.retriever.retrieve(query)
        except Exception:
            # Fallback: unfiltered retrieval
            nodes = self.index.as_retriever(similarity_top_k=5).retrieve(query)
            nodes = [n for n in nodes if n.metadata.get("content_type") == "table"]

        if not nodes:
            return "No tables found relevant to this query."

        tables_text = "\n\n".join(n.get_content() for n in nodes)
        prompt = (
            "You are a data analyst. Answer the question below using ONLY the table data provided.\n\n"
            f"QUESTION: {query}\n\n"
            f"TABLE DATA:\n{tables_text}\n\n"
            "ANSWER:"
        )
        response = self.llm.complete(prompt)
        return str(response)

    def __call__(self, query: str) -> str:
        return self.run(query)


# ── Tool registry ─────────────────────────────────────────────────────────────

def build_tools(index: VectorStoreIndex, llm) -> dict[str, Any]:
    """Return all tools keyed by name."""
    return {
        "retriever": RetrieverTool(index),
        "summarizer": SummarizerTool(llm),
        "table_analyst": TableAnalystTool(index, llm),
    }
