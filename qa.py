"""Answer questions with Claude, grounded in retrieved document chunks."""

from __future__ import annotations

import os
from typing import Iterator

import anthropic

from .ingest import Chunk
from .vectorstore import VectorStore

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question \
using ONLY the numbered context passages provided. Each passage is labeled with \
a source file and page number.

Rules:
- Ground every claim in the passages. Do not use outside knowledge.
- Cite the passages you rely on inline using their number, e.g. [1] or [2][3].
- If the passages do not contain the answer, say so plainly:
  "I couldn't find that in the provided documents." Do not guess.
- Be concise and direct. Quote short phrases when precision matters.
"""


def _format_context(hits: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks into a numbered block Claude can cite."""
    blocks = []
    for i, (chunk, _score) in enumerate(hits, start=1):
        header = f"[{i}] Source: {chunk.source} (page {chunk.page})"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, hits: list[tuple[Chunk, float]]) -> list[dict]:
    context = _format_context(hits)
    user_content = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}"
    )
    return [{"role": "user", "content": user_content}]


def answer(
    question: str,
    store: VectorStore,
    client: anthropic.Anthropic | None = None,
    k: int = 4,
    model: str = MODEL,
) -> tuple[str, list[tuple[Chunk, float]]]:
    """Retrieve, then ask Claude. Returns (answer_text, retrieved_hits)."""
    client = client or anthropic.Anthropic()
    hits = store.search(question, k=k)
    if not hits:
        return ("No documents have been indexed yet. Upload a PDF first.", [])

    response = client.messages.create(
        model=model,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        output_config={"effort": "medium"},
        messages=build_messages(question, hits),
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, hits


def answer_stream(
    question: str,
    store: VectorStore,
    client: anthropic.Anthropic | None = None,
    k: int = 4,
    model: str = MODEL,
) -> tuple[Iterator[str], list[tuple[Chunk, float]]]:
    """Streaming variant. Returns (text_chunk_iterator, retrieved_hits).

    Retrieval happens up front so the caller can show sources immediately;
    the iterator yields answer text as Claude generates it.
    """
    client = client or anthropic.Anthropic()
    hits = store.search(question, k=k)

    def _empty() -> Iterator[str]:
        yield "No documents have been indexed yet. Upload a PDF first."

    if not hits:
        return _empty(), []

    def _generate() -> Iterator[str]:
        with client.messages.stream(
            model=model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            output_config={"effort": "medium"},
            messages=build_messages(question, hits),
        ) as stream:
            yield from stream.text_stream

    return _generate(), hits
