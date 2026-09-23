"""Application runtime for document ingestion and RAG queries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.chat import ChatClient
from app.chunking import chunk_pages
from app.config import Settings
from app.document_loader import load_document
from app.embeddings import EmbeddingClient
from app.models import QueryResponse
from app.rag import RAGService
from app.retriever import VectorRetriever
from app.vector_store import FaissVectorStore


_EMBEDDING_BATCH_SIZE = 64


@dataclass(frozen=True, slots=True)
class IngestResult:
    document_id: str
    file_name: str
    page_count: int
    chunk_count: int
    generation: str
    already_indexed: bool


class RAGRuntime:
    """Own the loaded index and its clients for one application process."""

    def __init__(
        self,
        settings: Settings,
        embedder: Any,
        chat_client: Any,
        vector_store: FaissVectorStore,
    ) -> None:
        self.settings = settings
        self.index_dir = Path(settings.index_dir)
        self.embedder = embedder
        self.chat_client = chat_client
        self.vector_store = vector_store
        self._wire_pipeline()

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGRuntime":
        """Create provider clients and restore the last persisted index snapshot."""

        if not settings.api_key:
            raise ValueError("Set TRACERAG_API_KEY or OPENAI_API_KEY before using RAG.")
        index_dir = Path(settings.index_dir)
        if (index_dir / "CURRENT").is_file():
            store = FaissVectorStore.load(index_dir)
        else:
            store = FaissVectorStore()
        embedder = EmbeddingClient(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.embedding_model,
        )
        chat_client = ChatClient(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.chat_model,
        )
        return cls(settings, embedder, chat_client, store)

    def _wire_pipeline(self) -> None:
        self.retriever = VectorRetriever(self.embedder, self.vector_store)
        self.rag = RAGService(
            self.retriever,
            self.chat_client,
            reject_threshold=self.settings.reject_threshold,
        )

    def ingest(self, file_name: str, data: bytes) -> IngestResult:
        """Parse, chunk, embed, persist, and add one supported document."""

        pages = load_document(file_name, data)
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("No extractable text was found in the uploaded document.")

        document_id = pages[0].document_id
        existing_ids = {chunk.chunk_id for chunk in self.vector_store.chunks}
        pending_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
        if not pending_chunks:
            generation = (self.index_dir / "CURRENT").read_text(encoding="ascii").strip()
            return IngestResult(
                document_id, pages[0].file_name, len(pages), len(chunks), generation, True
            )

        batches = [
            pending_chunks[start : start + _EMBEDDING_BATCH_SIZE]
            for start in range(0, len(pending_chunks), _EMBEDDING_BATCH_SIZE)
        ]
        vectors = np.concatenate(
            [self.embedder.embed_texts([chunk.content for chunk in batch]) for batch in batches],
            axis=0,
        )

        candidate_store = (
            FaissVectorStore.load(self.index_dir)
            if (self.index_dir / "CURRENT").is_file()
            else FaissVectorStore()
        )
        candidate_store.add(pending_chunks, vectors)
        generation = candidate_store.save(self.index_dir)
        self.vector_store = candidate_store
        self._wire_pipeline()
        return IngestResult(
            document_id,
            pages[0].file_name,
            len(pages),
            len(pending_chunks),
            generation,
            False,
        )

    def query(self, query: str, top_k: int = 5) -> QueryResponse:
        """Retrieve and answer through the currently loaded index snapshot."""

        return self.rag.query(query, top_k=top_k)
