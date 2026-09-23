"""Query embeddings and retrieve source-aware top-K chunks from FAISS."""

from __future__ import annotations

from typing import Protocol

from app.embeddings import EmbeddingClient
from app.models import RetrievalResult
from app.vector_store import FaissVectorStore


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]): ...


class VectorRetriever:
    """A small retrieval interface that keeps embedding and index dimensions aligned."""

    def __init__(self, embedder: EmbeddingClient | TextEmbedder, store: FaissVectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Embed a non-empty query and return cosine-ranked chunks and scores."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if self._store.count == 0:
            return []

        vectors = self._embedder.embed_texts([query])
        matches = self._store.search(vectors[0], top_k)
        return [RetrievalResult(chunk=chunk, score=score) for chunk, score in matches]
