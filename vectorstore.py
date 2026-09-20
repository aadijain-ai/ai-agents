"""A tiny persistent vector store backed by NumPy.

Cosine similarity over a normalized matrix is more than fast enough for the
document counts a demo Q&A app handles, and it avoids native-build headaches
(FAISS) on Windows. Swap in FAISS/Chroma later if you outgrow it.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .embeddings import embed, embed_one
from .ingest import Chunk


class VectorStore:
    def __init__(self, dim: int | None = None):
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # (n, dim), L2-normalized
        self.dim = dim

    # ------------------------------------------------------------------ #
    # Building
    # ------------------------------------------------------------------ #
    def add(self, chunks: list[Chunk]) -> None:
        """Embed and append chunks to the index."""
        if not chunks:
            return
        vectors = embed([c.text for c in chunks])
        if self.dim is None:
            self.dim = vectors.shape[1]
        self.chunks.extend(chunks)
        self._matrix = (
            vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        )

    @property
    def size(self) -> int:
        return len(self.chunks)

    def sources(self) -> list[str]:
        """Distinct document names currently indexed, in first-seen order."""
        seen: list[str] = []
        for c in self.chunks:
            if c.source not in seen:
                seen.append(c.source)
        return seen

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the top-k (chunk, similarity) pairs for a query string."""
        if self._matrix is None or self.size == 0:
            return []
        q = embed_one(query)                       # (dim,), normalized
        scores = self._matrix @ q                  # cosine similarity
        k = min(k, self.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]        # sort the k winners
        return [(self.chunks[i], float(scores[i])) for i in top]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "chunks": [asdict(c) for c in self.chunks],
            "matrix": self._matrix,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "VectorStore":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        store = cls(dim=payload["dim"])
        store.chunks = [Chunk(**c) for c in payload["chunks"]]
        store._matrix = payload["matrix"]
        return store
