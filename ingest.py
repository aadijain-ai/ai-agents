"""Load PDF files and split their text into overlapping chunks for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import BinaryIO

from pypdf import PdfReader


@dataclass
class Chunk:
    """A retrievable piece of a document, with enough metadata to cite it."""

    text: str
    source: str          # file name the chunk came from
    page: int            # 1-indexed page number
    chunk_id: int        # position within the document
    metadata: dict = field(default_factory=dict)


def _clean(text: str) -> str:
    """Collapse the ragged whitespace that PDF extraction tends to produce."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(file: BinaryIO | str, source: str) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) for a PDF.

    `file` may be a path or any binary file-like object (e.g. a Streamlit upload).
    """
    reader = PdfReader(file)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into ~chunk_size character windows that overlap by `overlap`.

    Splitting on character length (rather than tokens) keeps the dependency
    footprint small; the overlap preserves context that straddles a boundary.
    Boundaries are nudged to the nearest sentence/word break when possible.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # Try to end on a clean boundary rather than mid-word.
        if end < n:
            window = text[start:end]
            for sep in ("\n\n", ". ", "\n", " "):
                idx = window.rfind(sep)
                if idx > chunk_size * 0.5:  # only if the break isn't too early
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_pdf(
    file: BinaryIO | str,
    source: str,
    chunk_size: int = 1000,
    overlap: int = 150,
    start_chunk_id: int = 0,
) -> list[Chunk]:
    """Read a PDF and return a flat list of Chunk objects, page by page."""
    chunks: list[Chunk] = []
    cid = start_chunk_id
    for page_num, page_text in extract_pages(file, source):
        for piece in chunk_text(page_text, chunk_size, overlap):
            chunks.append(
                Chunk(text=piece, source=source, page=page_num, chunk_id=cid)
            )
            cid += 1
    return chunks
