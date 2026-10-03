"""负责文档入库、向量索引持久化和 RAG 查询的应用运行时。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.chat import ChatClient
from app.chunking import chunk_pages
from app.config import Settings
from app.document_loader import load_document
from app.embeddings import EmbeddingClient, SentenceTransformerEmbeddingClient
from app.models import (
    Chunk,
    EmbeddingIndexCompatibilityError,
    LegacyEmbeddingIndexError,
    QueryResponse,
)
from app.rag import RAGService
from app.retriever import BM25Retriever, HybridRetriever, VectorRetriever
from app.relevance import RelevancePolicy
from app.vector_store import FaissVectorStore


_EMBEDDING_BATCH_SIZE = 64
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """记录一次入库的处理结果，供 API 和界面准确展示状态。"""

    document_id: str
    file_name: str
    page_count: int
    chunk_count: int
    generation: str
    already_indexed: bool
    replaced_existing: bool = False


class RAGRuntime:
    """在单个应用进程中持有已校验的索引和模型客户端。"""

    def __init__(
        self,
        settings: Settings,
        embedder: Any,
        chat_client: Any,
        vector_store: FaissVectorStore,
    ) -> None:
        """注入配置、模型、聊天客户端和已校验索引，组装应用管线。"""
        self.settings = settings
        self.index_dir = Path(settings.index_dir)
        self.embedding_identity = settings.embedding_identity
        self.embedder = embedder
        self.chat_client = chat_client
        self.vector_store = vector_store
        self._ensure_embedding_identity()
        self._wire_pipeline()

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGRuntime":
        """按当前配置创建模型客户端，并只恢复身份匹配的索引快照。"""

        if not settings.api_key:
            raise ValueError("Set TRACERAG_API_KEY or OPENAI_API_KEY to use chat.")
        embedding_identity = settings.embedding_identity
        index_dir = Path(settings.index_dir)
        legacy_chunks = None
        if (index_dir / "CURRENT").is_file():
            try:
                store = FaissVectorStore.load(
                    index_dir, expected_embedding_identity=embedding_identity
                )
            except LegacyEmbeddingIndexError as exc:
                # 身份缺失时丢弃旧向量，只使用保存的 Chunk 文本重新编码。
                logger.warning("旧索引需要重新向量化：%s", exc)
                legacy_chunks = FaissVectorStore.load_chunks_for_reindex(index_dir)
                store = FaissVectorStore()
        else:
            store = FaissVectorStore()
        if settings.embedding_provider == "local":
            embedder = SentenceTransformerEmbeddingClient(
                model_path=settings.local_embedding_path,
                device=settings.embedding_device,
            )
        elif settings.embedding_provider == "openai":
            embedder = EmbeddingClient(
                api_key=settings.api_key,
                base_url=settings.base_url,
                proxy_url=settings.api_proxy_url,
                model=settings.embedding_model,
            )
        else:
            raise ValueError(
                "TRACERAG_EMBEDDING_PROVIDER must be 'openai' or 'local'."
            )
        chat_client = ChatClient(
            api_key=settings.api_key,
            base_url=settings.base_url,
            proxy_url=settings.api_proxy_url,
            model=settings.chat_model,
        )
        runtime = cls(settings, embedder, chat_client, store)
        if legacy_chunks:
            runtime._reindex_legacy_chunks(legacy_chunks)
        return runtime

    def _wire_pipeline(self) -> None:
        """按配置组合检索器；索引更新后重建稀疏表，避免查询旧文档。"""
        vector = VectorRetriever(self.embedder, self.vector_store)
        if self.settings.retrieval_strategy == "vector":
            self.retriever = vector
        else:
            sparse = BM25Retriever(self.vector_store.chunks)
            self.retriever = (
                sparse if self.settings.retrieval_strategy == "bm25"
                else HybridRetriever(vector, sparse, rrf_k=self.settings.rrf_k)
            )
        self.rag = RAGService(
            self.retriever,
            self.chat_client,
            reject_threshold=self.settings.reject_threshold,
            relevance_policy=RelevancePolicy(
                vector_min_score=self.settings.reject_threshold,
                bm25_min_score=self.settings.bm25_min_score,
            ),
        )

    def _ensure_embedding_identity(self) -> None:
        """在入库或查询前阻止未标记或身份不符的向量被继续使用。"""
        stored_identity = self.vector_store.embedding_identity
        if (self.vector_store.count and stored_identity is None) or (
            stored_identity is not None and stored_identity != self.embedding_identity
        ):
            raise EmbeddingIndexCompatibilityError(
                "Loaded vector index does not match the configured embedding model; "
                "select a new index directory and re-upload the documents."
            )

    def _reindex_legacy_chunks(self, legacy_chunks: list[Chunk]) -> None:
        """重建旧索引向量，并按同名文件保留最后追加的文档版本。"""
        latest_document_ids: dict[str, str] = {}
        for chunk in legacy_chunks:
            # 旧 Runtime 按完整文档顺序追加，因此最后出现的 document_id 是最新版本。
            latest_document_ids[chunk.file_name.casefold()] = chunk.document_id
        chunks = [
            chunk
            for chunk in legacy_chunks
            if latest_document_ids[chunk.file_name.casefold()] == chunk.document_id
        ]
        if not chunks:
            return

        batches = [
            chunks[start : start + _EMBEDDING_BATCH_SIZE]
            for start in range(0, len(chunks), _EMBEDDING_BATCH_SIZE)
        ]
        vectors = np.concatenate(
            [self.embedder.embed_texts([chunk.content for chunk in batch]) for batch in batches],
            axis=0,
        )
        candidate_store = FaissVectorStore()
        candidate_store.add(chunks, vectors)
        # save 会先写新 generation，再原子替换 CURRENT；原 v1 快照保留作回退。
        candidate_store.save(
            self.index_dir,
            embedding_identity=self.embedding_identity,
        )
        self.vector_store = candidate_store
        self._wire_pipeline()

    def ingest(self, file_name: str, data: bytes) -> IngestResult:
        """解析并索引文档；同名更新先从候选索引移除旧片段。"""

        self._ensure_embedding_identity()
        pages = load_document(file_name, data)
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("No extractable text was found in the uploaded document.")

        normalized_name = pages[0].file_name
        if (self.index_dir / "CURRENT").is_file():
            base_store = FaissVectorStore.load(
                self.index_dir,
                expected_embedding_identity=self.embedding_identity,
            )
        else:
            base_store = self.vector_store

        old_file_chunks = [
            chunk
            for chunk in base_store.chunks
            if chunk.file_name.casefold() == normalized_name.casefold()
        ]
        incoming_ids = {chunk.chunk_id for chunk in chunks}
        old_file_ids = {chunk.chunk_id for chunk in old_file_chunks}
        if (
            old_file_chunks
            and old_file_ids == incoming_ids
            and len(old_file_chunks) == len(chunks)
        ):
            # 完全相同的文档保持幂等，不重新计算向量或切换快照。
            generation = self._current_generation()
            return IngestResult(
                pages[0].document_id,
                normalized_name,
                len(pages),
                len(chunks),
                generation,
                True,
            )

        # 更新采用独立候选索引；磁盘快照提交成功后才替换运行时索引。
        candidate_store = base_store.without_file_name(normalized_name)
        existing_ids = {chunk.chunk_id for chunk in candidate_store.chunks}
        pending_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
        if not pending_chunks:
            return IngestResult(
                pages[0].document_id,
                normalized_name,
                len(pages),
                len(chunks),
                self._current_generation(),
                True,
            )

        batches = [
            pending_chunks[start : start + _EMBEDDING_BATCH_SIZE]
            for start in range(0, len(pending_chunks), _EMBEDDING_BATCH_SIZE)
        ]
        vectors = np.concatenate(
            [self.embedder.embed_texts([chunk.content for chunk in batch]) for batch in batches],
            axis=0,
        )

        candidate_store.add(pending_chunks, vectors)
        generation = candidate_store.save(
            self.index_dir,
            embedding_identity=self.embedding_identity,
        )
        self.vector_store = candidate_store
        self._wire_pipeline()
        return IngestResult(
            pages[0].document_id,
            normalized_name,
            len(pages),
            len(pending_chunks),
            generation,
            False,
            replaced_existing=bool(old_file_chunks),
        )

    def _current_generation(self) -> str:
        """读取 CURRENT 指针；空内存索引返回空字符串。"""
        pointer = self.index_dir / "CURRENT"
        if not pointer.is_file():
            return ""
        return pointer.read_text(encoding="ascii").strip()

    def query(self, query: str, top_k: int = 5) -> QueryResponse:
        """核对索引身份后执行检索、拒答判断和引用整理。"""

        self._ensure_embedding_identity()
        return self.rag.query(query, top_k=top_k)
