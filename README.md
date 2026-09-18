# Multimodal Agentic AI Research System

> An autonomous AI research agent that analyzes complex technical PDFs, tables, and visual charts using LangGraph stateful workflows and local LLMs via Ollama.

---

## 🚀 Demo

Launch locally with:
```bash
python app.py
```
Then open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 📌 Overview

This system is a multimodal agentic pipeline built for deep document research. Unlike simple RAG systems, it uses **LangGraph** to orchestrate stateful, multi-step reasoning — dynamically selecting tools, routing queries, and self-correcting when answers are incomplete or ambiguous.

### What makes it different from plain RAG?

| Feature | Plain RAG | This System |
|---|---|---|
| Tool selection | Single retriever | Dynamic router (text / table / summary) |
| Error handling | None | Self-correction loop with query refinement |
| Multimodal | Text only | Text + Tables + Charts |
| State | Stateless | LangGraph stateful graph |

---

## ✨ Features

- 🤖 **Autonomous agent** with dynamic tool selection and query routing
- 🔄 **Self-correction loops** — agent retries with a refined query when confidence is low
- 📊 **Multimodal parsing** — text, embedded tables (via pdfplumber), and chart images (via PyMuPDF)
- 📚 **LlamaIndex** for advanced document indexing and retrieval
- 🔒 **Fully local inference** via Ollama — no API keys, no data leaves your machine
- ⚡ **Structured answers** with verifiable source citations
- 🖥️ **Clean Gradio interface** for interactive research queries

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agentic Framework | LangGraph |
| Document Indexing | LlamaIndex |
| Local LLM | Ollama (`llama3` default) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector Store | FAISS (via LlamaIndex) |
| PDF Parsing | PyMuPDF + pdfplumber |
| Interface | Gradio |
| Language | Python 3.10+ |

---

## 📂 Project Structure

```
agentic-research/
├── app.py                      # Gradio interface & event wiring
├── agent/
│   ├── __init__.py
│   ├── graph.py                # LangGraph state graph (route→execute→correct→answer)
│   ├── tools.py                # Tool definitions (RetrieverTool, SummarizerTool, TableAnalystTool)
│   ├── router.py               # Query routing logic (heuristic + LLM-assisted)
│   └── corrector.py            # Self-correction loop with confidence scoring
├── indexing/
│   ├── __init__.py
│   ├── pdf_parser.py           # Multimodal PDF parsing (text + tables + charts)
│   └── index_builder.py        # LlamaIndex pipeline + persistence
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

### 1. Clone & install

```bash
git clone https://github.com/yourusername/agentic-research.git
cd agentic-research

pip install -r requirements.txt
```

### 2. Pull Ollama models

```bash
# Main LLM
ollama pull llama3

# Embedding model
ollama pull nomic-embed-text
```

> **Tip:** You can use any Ollama-compatible model. Set `OLLAMA_MODEL` and `OLLAMA_EMBED_MODEL` env vars to override defaults.

### 3. Run the app

```bash
python app.py
```

Then visit [http://localhost:7860](http://localhost:7860).

---

## 🧠 How It Works

```
User uploads PDF
       │
       ▼
[pdf_parser.py] ──────────────────────────────────────────
  • PyMuPDF   → extracts text per page + chart images
  • pdfplumber → extracts structured tables as DataFrames
       │
       ▼
[index_builder.py]
  • Creates LlamaIndex Documents (text / table / chart nodes)
  • Builds FAISS vector index with Ollama embeddings
  • Persists index to disk for fast reload
       │
       ▼
User submits research query
       │
       ▼
[graph.py] — LangGraph pipeline
  ┌──────────────────────────────────────┐
  │  route → execute → correct → answer  │
  └──────────────────────────────────────┘
       │
  route (router.py)
    • Keyword heuristics → fast routing
    • LLM-assisted routing for ambiguous queries
    • Selects: retriever | table_analyst | summarizer
       │
  execute (tools.py)
    • RetrieverTool    → semantic vector search
    • TableAnalystTool → table-specific retrieval + analysis
    • SummarizerTool   → passage summarization
       │
  correct (corrector.py)
    • Synthesizes answer from retrieved context
    • Confidence scoring (length + uncertainty patterns)
    • Refines query and retries (up to 2x) if low confidence
       │
  answer (graph.py)
    • Polishes final response
    • Extracts page citation references
       │
       ▼
Structured answer + citations returned to Gradio UI
```

---

## 🔧 Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3` | LLM for reasoning and answer synthesis |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for vector search |
| `PORT` | `7860` | Gradio server port |

---

## 🔮 Roadmap

- [ ] Deploy on Hugging Face Spaces
- [ ] Add web search tool to the agent
- [ ] Support multi-document research sessions
- [ ] Add memory persistence across sessions
- [ ] Vision model integration for true chart understanding vision model

---

## 📄 License

MIT License
