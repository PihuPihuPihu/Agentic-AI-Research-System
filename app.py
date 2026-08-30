"""
app.py — Gradio interface for the Multimodal Agentic AI Research System.

Workflow:
  1. User uploads a PDF
  2. PDF is parsed (text + tables + charts)
  3. LlamaIndex builds a vector index
  4. User submits research queries
  5. LangGraph agent routes, retrieves, self-corrects, and returns citations
"""

import os
import time
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from indexing.pdf_parser import parse_pdf
from indexing.index_builder import build_index
from agent.graph import build_graph, run_query

# ── Global state ──────────────────────────────────────────────────────────────

_state = {
    "graph": None,
    "parsed_doc": None,
    "index": None,
    "pdf_name": None,
}

UPLOAD_DIR = Path("uploads")
INDEX_DIR = Path("index_cache")
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

MODEL = os.getenv("OLLAMA_MODEL", "llama3")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


# ── Backend functions ─────────────────────────────────────────────────────────

def process_pdf(pdf_file, progress=gr.Progress()):
    """Parse the uploaded PDF and build the agent graph."""
    if pdf_file is None:
        return (
            gr.update(value="⚠️ Please upload a PDF first.", visible=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    try:
        progress(0.1, desc="Parsing PDF...")
        # Copy uploaded file to a stable path
        dest = UPLOAD_DIR / Path(pdf_file.name).name
        shutil.copy(pdf_file.name, dest)

        parsed_doc = parse_pdf(str(dest), output_dir=str(UPLOAD_DIR / "assets"))
        _state["parsed_doc"] = parsed_doc
        _state["pdf_name"] = parsed_doc.title

        progress(0.5, desc="Building vector index...")
        persist_path = str(INDEX_DIR / parsed_doc.title)
        index = build_index(
            parsed_doc,
            persist_dir=persist_path,
            model=MODEL,
            embed_model=EMBED_MODEL,
        )
        _state["index"] = index

        progress(0.85, desc="Initialising agent...")
        graph = build_graph(index, model=MODEL)
        _state["graph"] = graph

        progress(1.0, desc="Ready!")

        stats = (
            f"✅ **{parsed_doc.title}** loaded successfully\n\n"
            f"- 📄 Pages: {parsed_doc.total_pages}\n"
            f"- 📊 Tables: {sum(len(p.tables) for p in parsed_doc.pages)}\n"
            f"- 🖼️ Charts: {sum(len(p.chart_paths) for p in parsed_doc.pages)}"
        )
        return (
            gr.update(value=stats, visible=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    except Exception as exc:
        return (
            gr.update(value=f"❌ Error processing PDF: {exc}", visible=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )


def answer_query(query: str, history: list):
    """Run the query through the agent and stream the result."""
    if not query.strip():
        return history, ""

    if _state["graph"] is None:
        history.append((query, "⚠️ Please upload and process a PDF first."))
        return history, ""

    # Show thinking indicator
    history.append((query, "🔍 Researching…"))
    yield history, ""

    try:
        result = run_query(_state["graph"], query)

        answer = result.get("answer", "No answer generated.")
        citations = result.get("citations", [])
        attempts = result.get("attempts", 1)
        elapsed = result.get("elapsed_ms", 0)
        tool_used = result.get("tool_name", "retriever")
        error = result.get("error")

        confidence = result.get("confidence", 0.0)
        conf_bar = "🟢" if confidence >= 0.75 else "🟡" if confidence >= 0.5 else "🔴"

        if error:
            response = f"❌ Agent error: {error}"
        else:
            citation_str = ", ".join(citations) if citations else "N/A"
            reroute_note = f" *(rerouted ×{attempts - 1})*" if attempts > 1 else ""
            response = (
                f"{answer}\n\n"
                f"---\n"
                f"📎 **Sources:** {citation_str} &nbsp;|&nbsp; "
                f"🔧 **Tool:** `{tool_used}`{reroute_note} &nbsp;|&nbsp; "
                f"{conf_bar} **Confidence:** {int(confidence * 100)}% &nbsp;|&nbsp; "
                f"⏱️ **Time:** {elapsed}ms"
            )

        history[-1] = (query, response)

    except Exception as exc:
        history[-1] = (query, f"❌ Unexpected error: {exc}")

    yield history, ""


def clear_chat():
    return [], ""


def reset_document():
    _state["graph"] = None
    _state["parsed_doc"] = None
    _state["index"] = None
    _state["pdf_name"] = None
    return (
        gr.update(value=None),
        gr.update(value="Upload a PDF to begin.", visible=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        [],
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
#subtitle { text-align: center; color: #6b7280; margin-bottom: 1.5rem; }
#status-box { border-radius: 8px; padding: 0.75rem 1rem; background: #f9fafb; }
.chat-bubble { font-size: 0.95rem; }
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:

    # ── Header ─────────────────────────────────────────────────────────────
    gr.Markdown("# 🔬 Multimodal Agentic AI Research System", elem_id="title")
    gr.Markdown(
        "Upload a technical PDF and ask research questions. "
        "The agent autonomously selects tools, retrieves evidence, "
        "and self-corrects to deliver cited answers.",
        elem_id="subtitle",
    )

    with gr.Row():
        # ── Left panel: document upload ────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📁 Document")
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                file_count="single",
            )
            process_btn = gr.Button("⚡ Process Document", variant="primary")
            reset_btn = gr.Button("🗑️ Reset", variant="secondary")

            status_box = gr.Markdown(
                value="Upload a PDF to begin.",
                visible=True,
                elem_id="status-box",
            )

            gr.Markdown("### ⚙️ Model Info")
            gr.Markdown(
                f"**LLM:** `{MODEL}`  \n"
                f"**Embeddings:** `{EMBED_MODEL}`  \n"
                "*(configured via env vars)*\n\n"
                "**Agent pipeline:**  \n"
                "`route → execute → correct → answer`  \n\n"
                "🔄 Auto-reroutes on low confidence  \n"
                "📎 Verifiable page citations  \n"
                "⚡ Target latency: < 4 s"
            )

            gr.Markdown("### 💡 Example Queries")
            examples = gr.Examples(
                examples=[
                    ["What are the main findings of this paper?"],
                    ["Summarize the methodology section."],
                    ["What do the tables show about the results?"],
                    ["What conclusions does the author draw?"],
                    ["Are there any charts or figures? What do they depict?"],
                ],
                inputs=[],  # will be wired below
                label="",
            )

        # ── Right panel: chat interface ────────────────────────────────────
        with gr.Column(scale=3):
            gr.Markdown("### 💬 Research Chat")
            chatbot = gr.Chatbot(
                label="",
                height=480,
                elem_classes=["chat-bubble"],
                show_label=False,
                bubble_full_width=False,
            )
            with gr.Row():
                query_input = gr.Textbox(
                    placeholder="Ask a research question about the document…",
                    show_label=False,
                    scale=5,
                    interactive=False,
                )
                submit_btn = gr.Button("Ask →", variant="primary", scale=1, interactive=False)
            clear_btn = gr.Button("Clear Chat", size="sm")

    # ── Event wiring ───────────────────────────────────────────────────────

    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[status_box, query_input, submit_btn],
    )

    submit_btn.click(
        fn=answer_query,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input],
    )

    query_input.submit(
        fn=answer_query,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input],
    )

    clear_btn.click(fn=clear_chat, outputs=[chatbot, query_input])

    reset_btn.click(
        fn=reset_document,
        outputs=[pdf_input, status_box, query_input, submit_btn, chatbot],
    )

    # Wire example queries into the text box
    for ex_btn in (examples.dataset if hasattr(examples, "dataset") else []):
        pass  # Examples auto-populate via gr.Examples


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
    )
