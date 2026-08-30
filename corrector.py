"""
corrector.py — Self-correction loop for the agent.

On low-confidence answers the loop:
  1. Refines the query text (better phrasing)
  2. Reroutes to a different tool (strategy switch)
  3. Retries up to max_retries times before returning best answer
"""

from __future__ import annotations
import re
from typing import Callable


# ── Confidence evaluator ──────────────────────────────────────────────────────

LOW_CONFIDENCE_PATTERNS = re.compile(
    r"\b(i don'?t know|not sure|cannot find|no information|unclear|"
    r"i (am|'m) unable|no relevant|not found|insufficient|sorry|"
    r"unable to determine|no data|not mentioned|not available)\b",
    re.IGNORECASE,
)

# Score thresholds
MIN_ANSWER_LENGTH = 40
HIGH_CONFIDENCE_LENGTH = 150


def confidence_score(answer: str) -> float:
    """
    Return a score 0.0–1.0 representing answer confidence.
      < 0.5  → low confidence, trigger retry+reroute
      >= 0.5 → acceptable
    """
    text = answer.strip()
    if not text:
        return 0.0

    score = 1.0

    # Penalise short answers
    if len(text) < MIN_ANSWER_LENGTH:
        score -= 0.6
    elif len(text) < HIGH_CONFIDENCE_LENGTH:
        score -= 0.2

    # Penalise uncertainty language
    hits = len(LOW_CONFIDENCE_PATTERNS.findall(text))
    score -= hits * 0.25

    return max(0.0, min(1.0, score))


def is_low_confidence(answer: str, threshold: float = 0.5) -> bool:
    return confidence_score(answer) < threshold


# ── Query refiner ─────────────────────────────────────────────────────────────

def refine_query(original_query: str, previous_answer: str, llm) -> str:
    """Ask the LLM to produce a better-phrased query given the failed attempt."""
    prompt = (
        "You are a research assistant refining a failed document search query.\n\n"
        f"Original question: \"{original_query}\"\n"
        f"Previous (insufficient) answer: \"{previous_answer}\"\n\n"
        "Write a DIFFERENT, more specific search query that is likely to retrieve "
        "better information. Output ONLY the refined query, no explanation:"
    )
    try:
        refined = str(llm.complete(prompt)).strip()
        return refined if refined else original_query
    except Exception:
        return original_query


# ── Tool rerouter ─────────────────────────────────────────────────────────────

# On each retry attempt, fall back to the next tool in priority order.
_REROUTE_FALLBACK: dict[str, list[str]] = {
    "retriever":     ["summarizer", "table_analyst"],
    "table_analyst": ["retriever",  "summarizer"],
    "summarizer":    ["retriever",  "table_analyst"],
}


def reroute_tool(current_tool: str, attempt: int) -> str:
    """
    Return the next tool to try on the given retry attempt (1-indexed).
    Falls back gracefully if the index is out of range.
    """
    fallbacks = _REROUTE_FALLBACK.get(current_tool, ["retriever", "summarizer"])
    idx = (attempt - 1) % len(fallbacks)
    return fallbacks[idx]


# ── Self-correction orchestrator ──────────────────────────────────────────────

def self_correct(
    query: str,
    tool_fn: Callable[[str], str],
    synthesize_fn: Callable[[str, str], str],
    llm,
    tools: dict | None = None,
    original_tool: str = "retriever",
    max_retries: int = 2,
) -> tuple[str, int, str]:
    """
    Run tool_fn, evaluate confidence, and retry+reroute when confidence is low.

    On each retry:
      • Refines the query text via the LLM
      • Switches to a different tool (reroutes) for a fresh retrieval strategy

    Args:
        query:         Original user query.
        tool_fn:       Initial tool callable.
        synthesize_fn: Function(query, context) → answer string.
        llm:           Ollama LLM instance for refinement.
        tools:         Dict of all available tools (enables rerouting).
        original_tool: Name of the initial tool selected by router.
        max_retries:   Maximum number of retry+reroute attempts.

    Returns:
        (final_answer, total_attempts, final_tool_used)
    """
    current_query = query
    current_tool_name = original_tool
    current_tool_fn = tool_fn
    best_answer = ""
    best_score = -1.0

    for attempt in range(1, max_retries + 2):  # +2: initial + retries
        raw_context = current_tool_fn(current_query)
        answer = synthesize_fn(current_query, raw_context)
        score = confidence_score(answer)

        # Track best answer across all attempts
        if score > best_score:
            best_score = score
            best_answer = answer

        if not is_low_confidence(answer):
            return answer, attempt, current_tool_name

        if attempt <= max_retries:
            # 1. Refine the query
            current_query = refine_query(query, answer, llm)

            # 2. Reroute to a different tool
            next_tool_name = reroute_tool(current_tool_name, attempt)
            if tools and next_tool_name in tools:
                current_tool_fn = tools[next_tool_name]
                current_tool_name = next_tool_name
            # else: stay on same tool with refined query only

            print(
                f"[Corrector] 🔄 Retry {attempt}/{max_retries} — "
                f"tool: '{current_tool_name}' | "
                f"refined query: '{current_query[:60]}...'"
            )

    # Return the highest-confidence answer found across all attempts
    return best_answer, max_retries + 1, current_tool_name
