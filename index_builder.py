"""
index_builder.py — Builds a LlamaIndex VectorStoreIndex from a ParsedDocument.
Handles text chunks, table markdown, and chart image descriptions separately.
"""

import os
from pathlib import Path
from typing import Optional

from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from indexing.pdf_parser import ParsedDocument


# ── Global LlamaIndex settings ──────────────────────────────────────────────

def configure_llama_index(model: str = "llama3", embed_model: str = "nomic-embed-text"):
    """Configure LlamaIndex to use local Ollama models."""
    Settings.llm = Ollama(model=model, request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name=embed_model)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)


# ── Document builder ─────────────────────────────────────────────────────────

def build_documents(parsed_doc: ParsedDocument) -> list[Document]:
    """
    Convert a ParsedDocument into LlamaIndex Document objects.
    Each page's text, tables, and chart placeholders become separate nodes
    so retrieval can pinpoint exactly which content answered the query.
    """
    documents = []

    for page in parsed_doc.pages:
        base_meta = {
            "source": parsed_doc.file_path,
            "title": parsed_doc.title,
            "page": page.page_number,
        }

        # 1. Text chunk
        if page.text.strip():
            documents.append(Document(
                text=page.text,
                metadata={**base_meta, "content_type": "text"},
                doc_id=f"{parsed_doc.title}_p{page.page_number}_text",
            ))

        # 2. Table chunks (one Document per table, as Markdown)
        for i, df in enumerate(page.tables):
            md = df.to_markdown(index=False)
            documents.append(Document(
                text=f"[Table on page {page.page_number}, index {i}]\n\n{md}",
                metadata={**base_meta, "content_type": "table", "table_index": i},
                doc_id=f"{parsed_doc.title}_p{page.page_number}_table{i}",
            ))

        # 3. Chart references (placeholder text — vision model not required)
        for j, chart_path in enumerate(page.chart_paths):
            documents.append(Document(
                text=(
                    f"[Figure/Chart on page {page.page_number}, index {j}]\n"
                    f"Image path: {chart_path}\n"
                    "This is a visual element extracted from the PDF. "
                    "Refer to the image file for details."
                ),
                metadata={**base_meta, "content_type": "chart", "chart_path": chart_path},
                doc_id=f"{parsed_doc.title}_p{page.page_number}_chart{j}",
            ))

    return documents


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(
    parsed_doc: ParsedDocument,
    persist_dir: Optional[str] = None,
    model: str = "llama3",
    embed_model: str = "nomic-embed-text",
) -> VectorStoreIndex:
    """
    Build (or load from cache) a VectorStoreIndex for the given ParsedDocument.
    """
    configure_llama_index(model=model, embed_model=embed_model)

    if persist_dir and Path(persist_dir).exists():
        print(f"[Indexer] Loading existing index from '{persist_dir}'...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        print("[Indexer] ✅ Index loaded from cache.")
        return index

    print(f"[Indexer] Building index for '{parsed_doc.title}'...")
    documents = build_documents(parsed_doc)
    print(f"[Indexer] Created {len(documents)} document nodes.")

    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    if persist_dir:
        index.storage_context.persist(persist_dir=persist_dir)
        print(f"[Indexer] ✅ Index persisted to '{persist_dir}'.")

    return index


# ── Retriever factory ─────────────────────────────────────────────────────────

def get_retriever(index: VectorStoreIndex, top_k: int = 5):
    """Return a retriever that fetches the top-k most relevant nodes."""
    return index.as_retriever(similarity_top_k=top_k)
