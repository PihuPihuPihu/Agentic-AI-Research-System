"""
pdf_parser.py — Multimodal PDF parsing: text, tables, and charts.
Uses PyMuPDF for text/images and pdfplumber for table extraction.
"""

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from PIL import Image
import io
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParsedPage:
    page_number: int
    text: str
    tables: List[pd.DataFrame] = field(default_factory=list)
    chart_paths: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    file_path: str
    title: str
    pages: List[ParsedPage] = field(default_factory=list)
    total_pages: int = 0

    def all_text(self) -> str:
        return "\n\n".join(
            f"[Page {p.page_number}]\n{p.text}" for p in self.pages
        )

    def all_tables(self) -> List[dict]:
        tables = []
        for page in self.pages:
            for i, df in enumerate(page.tables):
                tables.append({
                    "page": page.page_number,
                    "table_index": i,
                    "markdown": df.to_markdown(index=False),
                    "dataframe": df,
                })
        return tables

    def all_charts(self) -> List[str]:
        return [
            path
            for page in self.pages
            for path in page.chart_paths
        ]


def _is_likely_chart(image: fitz.Pixmap, min_size: int = 100) -> bool:
    """Heuristic: images larger than min_size x min_size are probably charts."""
    return image.width >= min_size and image.height >= min_size


def parse_pdf(file_path: str, output_dir: str = "parsed_assets") -> ParsedDocument:
    """
    Full multimodal parse of a PDF file.
    Returns a ParsedDocument containing text, tables, and saved chart image paths.
    """
    path = Path(file_path)
    output_dir = Path(output_dir) / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = ParsedDocument(
        file_path=file_path,
        title=path.stem,
    )

    # --- Text + Charts via PyMuPDF ---
    pdf_fitz = fitz.open(file_path)
    doc.total_pages = len(pdf_fitz)

    fitz_pages: dict[int, ParsedPage] = {}

    for page_num, page in enumerate(pdf_fitz, start=1):
        text = page.get_text("text").strip()
        chart_paths = []

        # Extract embedded images (charts/figures)
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = pdf_fitz.extract_image(xref)
                image_bytes = base_image["image"]
                pixmap = fitz.Pixmap(pdf_fitz, xref)

                if _is_likely_chart(pixmap):
                    img_filename = output_dir / f"page{page_num}_chart{img_index}.png"
                    if pixmap.n >= 5:  # CMYK or other, convert to RGB
                        pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                    pixmap.save(str(img_filename))
                    chart_paths.append(str(img_filename))
            except Exception:
                pass  # Skip corrupt/unreadable images

        parsed_page = ParsedPage(
            page_number=page_num,
            text=text,
            chart_paths=chart_paths,
            metadata={"width": page.rect.width, "height": page.rect.height},
        )
        fitz_pages[page_num] = parsed_page

    pdf_fitz.close()

    # --- Tables via pdfplumber ---
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_tables = page.extract_tables()
            dataframes = []
            for raw_table in raw_tables:
                if raw_table and len(raw_table) > 1:
                    try:
                        # First row as header
                        header = raw_table[0]
                        rows = raw_table[1:]
                        # Clean None values
                        header = [str(h) if h else f"Col{i}" for i, h in enumerate(header)]
                        rows = [[str(cell) if cell else "" for cell in row] for row in rows]
                        df = pd.DataFrame(rows, columns=header)
                        dataframes.append(df)
                    except Exception:
                        pass
            if page_num in fitz_pages:
                fitz_pages[page_num].tables = dataframes

    doc.pages = [fitz_pages[k] for k in sorted(fitz_pages)]

    # Save a summary manifest
    manifest = {
        "file": file_path,
        "title": doc.title,
        "total_pages": doc.total_pages,
        "total_tables": sum(len(p.tables) for p in doc.pages),
        "total_charts": sum(len(p.chart_paths) for p in doc.pages),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Parser] ✅ Parsed '{doc.title}': "
          f"{doc.total_pages} pages, "
          f"{manifest['total_tables']} tables, "
          f"{manifest['total_charts']} charts")

    return doc
