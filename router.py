"""
router.py — Determines which tool should handle a given query.
Uses keyword heuristics + LLM-based routing for ambiguous cases.
"""

from __future__ import annotations
import re


# ── Heuristic patterns ────────────────────────────────────────────────────────

TABLE_KEYWORDS = re.compile(
    r"\b(table|row|column|cell|data|value|figure|statistic|number|count|"
    r"percent|average|mean|sum|total|compare|trend|chart|graph)\b",
    re.IGNORECASE,
)

SUMMARY_KEYWORDS = re.compile(
    r"\b(summarize|summary|overview|brief|explain|describe|what is|what are|"
    r"outline|abstract|gist|main point|key point|conclusion)\b",
    re.IGNORECASE,
)


def route_query(query: str, llm=None) -> str:
    """
    Decide which tool should answer the query.

    Returns one of: 'retriever', 'table_analyst', 'summarizer'
    """
    q = query.strip()

    # Fast heuristic routing
    if TABLE_KEYWORDS.search(q):
        return "table_analyst"
    if SUMMARY_KEYWORDS.search(q):
        return "summarizer"

    # Default: semantic retrieval
    if llm is None:
        return "retriever"

    # LLM-assisted routing for ambiguous queries
    prompt = (
        "You are a query router for a document research system. "
        "Given the user's question, choose the BEST tool to answer it.\n\n"
        "Tools:\n"
        "- retriever: finds relevant text passages from the document\n"
        "- table_analyst: answers questions about numbers, tables, or data\n"
        "- summarizer: provides summaries or overviews of document content\n\n"
        f"User question: \"{q}\"\n\n"
        "Respond with ONLY the tool name (retriever / table_analyst / summarizer):"
    )
    try:
        response = str(llm.complete(prompt)).strip().lower()
        if response in ("retriever", "table_analyst", "summarizer"):
            return response
    except Exception:
        pass

    return "retriever"
