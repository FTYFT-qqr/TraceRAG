"""使用 FAISS 保存向量，并用 JSON 快照映射回可追溯文本片段。"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

import faiss
import numpy as np

from app.models import (
    Chunk,
    EmbeddingIdentity,
    EmbeddingIndexCompatibilityError,
    EmbeddingModelMismatchError,
    LegacyEmbeddingIndexError,
)


_SNAPSHOT_VERSION = 3
_GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class FaissVectorStore:
    """维护有序文本片段与归一化 FAISS 向量之间的一一对应关系。"""

    def __init__(self, embedding_identity: EmbeddingIdentity | None = None) -> None:
        """创建空的 FAISS 存储，并可预先绑定向量身份。"""
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []
        self.embedding_identity = embedding_identity

    @property
    def count(self) -> int:
        """返回当前索引中保存的 Chunk 数量。"""
        return len(self._chunks)

    @property
    def dimension(self) -> int | None:
        """返回向量维度；索引为空时返回空值。"""
        return self._index.d if self._index is not None else None

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """以不可变元组暴露当前 Chunk 顺序，避免外部修改索引状态。"""
        return tuple(self._chunks)

    def search(self, embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        """返回相似度最高的片段及其来源信息。"""

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if self._index is None or not self._chunks:
            return []

        query = np.asarray(embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1 or query.shape[1] != self._index.d:
            raise ValueError("Query embedding dimension does not match the FAISS index.")
        if not np.isfinite(query).all() or np.linalg.norm(query) == 0:
            raise ValueError("Query embedding must contain finite non-zero values.")

        query = query.copy()
        faiss.normalize_L2(query)
        scores, indexes = self._index.search(query, min(top_k, self.count))
        results: list[tuple[Chunk, float]] = []
        for score, index in zip(scores[0], indexes[0]):
            if index < 0:
                continue
            results.append((self._chunks[int(index)], float(score)))
        return results

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """添加一组逐行对齐的片段和向量，并归一化以支持余弦检索。"""

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Expected one two-dimensional embedding vector per chunk.")
        if not chunks:
            if vectors.shape[0] != 0:
                raise ValueError("Cannot add vectors without chunks.")
            return
        if vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("Embedding vectors must have a positive dimension and finite values.")
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Embedding vectors cannot be zero vectors.")

        known_ids = {chunk.chunk_id for chunk in self._chunks}
        incoming_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(incoming_ids)) != len(incoming_ids) or known_ids.intersection(incoming_ids):
            raise ValueError("Chunk IDs must be unique in the vector store.")
        if self._index is not None and self._index.d != vectors.shape[1]:
            raise ValueError("Embedding dimension does not match the existing FAISS index.")

        vectors = vectors.copy()
        faiss.normalize_L2(vectors)
        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def without_file_name(self, file_name: str) -> "FaissVectorStore":
        """返回移除指定文件片段后的新索引，不修改当前实例。"""
        if self._index is None or not self._chunks:
            return FaissVectorStore(self.embedding_identity)

        target_name = file_name.casefold()
        keep_positions = [
            index
            for index, chunk in enumerate(self._chunks)
            if chunk.file_name.casefold() != target_name
        ]
        kept_chunks = [self._chunks[index] for index in keep_positions]
        candidate = FaissVectorStore(self.embedding_identity)
        if not kept_chunks:
            return candidate

        # 从 Flat 索引重建保留向量，确保替换文档时其他文件的向量不变。
        kept_vectors = np.vstack(
            [self._index.reconstruct(index) for index in keep_positions]
        ).astype(np.float32, copy=False)
        candidate.add(kept_chunks, kept_vectors)
        return candidate

    def without_document_id(self, document_id: str) -> "FaissVectorStore":
        """构建移除指定文档后的独立索引；空结果仍保留向量维度。"""
        positions = [
            position for position, chunk in enumerate(self._chunks)
            if chunk.document_id != document_id
        ]
        candidate = FaissVectorStore(self.embedding_identity)
        if self._index is None:
            return candidate
        if not positions:
            # 零行 Flat 索引仍可保存并恢复，不能删除 CURRENT 暴露旧知识库。
            candidate._index = faiss.IndexFlatIP(self._index.d)
            return candidate
        candidate.add(
            [self._chunks[position] for position in positions],
            np.vstack([self._index.reconstruct(position) for position in positions]),
        )
        return candidate

    def save(
        self,
        directory: str | Path,
        *,
        embedding_identity: EmbeddingIdentity,
    ) -> str:
        """写入不可变快照，身份元数据就绪后再原子切换 CURRENT。"""

        if self._index is None:
            raise ValueError("Cannot save an empty vector store.")
        if (
            self.embedding_identity is not None
            and self.embedding_identity != embedding_identity
        ):
            raise EmbeddingModelMismatchError(
                "Cannot save vectors under a different embedding identity; rebuild the index."
            )

        root = Path(directory)
        snapshots = root / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        temporary = snapshots / f".tmp-{generation}"
        destination = snapshots / generation
        temporary.mkdir()
        try:
            faiss.write_index(self._index, str(temporary / "index.faiss"))
            (temporary / "chunks.json").write_text(
                json.dumps([chunk.to_dict() for chunk in self._chunks], ensure_ascii=False),
                encoding="utf-8",
            )
            manifest = {
                "version": _SNAPSHOT_VERSION,
                "generation": generation,
                "count": self.count,
                "dimension": self.dimension,
                "embedding_identity": embedding_identity.to_dict(),
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, destination)
            pointer_tmp = root / f".CURRENT-{generation}.tmp"
            pointer_tmp.write_text(generation, encoding="ascii")
            os.replace(pointer_tmp, root / "CURRENT")
            # 仅在磁盘指针切换成功后更新内存身份，失败时保留旧状态。
            self.embedding_identity = embedding_identity
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return generation

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_embedding_identity: EmbeddingIdentity,
    ) -> "FaissVectorStore":
        """加载快照并拒绝旧格式或与当前模型不一致的索引。"""

        root = Path(directory)
        try:
            generation = (root / "CURRENT").read_text(encoding="ascii").strip()
        except OSError as exc:
            raise FileNotFoundError(f"No saved vector index at {root}.") from exc
        if not _GENERATION_PATTERN.fullmatch(generation):
            raise ValueError("Vector index CURRENT pointer is invalid.")

        snapshot = root / "snapshots" / generation
        try:
            manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
            chunk_data = json.loads((snapshot / "chunks.json").read_text(encoding="utf-8"))
            index = faiss.read_index(str(snapshot / "index.faiss"))
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            raise ValueError("Saved vector index snapshot is incomplete or unreadable.") from exc

        if manifest.get("version") != _SNAPSHOT_VERSION:
            raise LegacyEmbeddingIndexError(
                "Saved vector index has no compatible embedding identity; "
                "its saved text must be re-embedded before use."
            )
        try:
            stored_identity = EmbeddingIdentity.from_dict(
                manifest.get("embedding_identity")
            )
        except ValueError as exc:
            raise LegacyEmbeddingIndexError(
                "Saved vector index is missing a valid embedding identity; "
                "its saved text must be re-embedded before use."
            ) from exc
        if stored_identity != expected_embedding_identity:
            raise EmbeddingModelMismatchError(
                "Configured embedding model does not match the saved vector index; "
                "use the original model or select a new index directory."
            )

        if (
            manifest.get("generation") != generation
            or manifest.get("count") != len(chunk_data)
            or manifest.get("dimension") != index.d
            or index.ntotal != len(chunk_data)
        ):
            raise ValueError("Saved vector index metadata does not match its FAISS index.")

        store = cls(stored_identity)
        store._index = index
        store._chunks = [Chunk.from_dict(item) for item in chunk_data]
        if len({chunk.chunk_id for chunk in store._chunks}) != len(store._chunks):
            raise ValueError("Saved vector index contains duplicate chunk IDs.")
        return store

    @classmethod
    def load_chunks_for_reindex(cls, directory: str | Path) -> list[Chunk]:
        """校验旧快照结构后只取文本元数据，绝不复用其未知来源向量。"""
        root = Path(directory)
        try:
            generation = (root / "CURRENT").read_text(encoding="ascii").strip()
        except OSError as exc:
            raise FileNotFoundError(f"No saved vector index at {root}.") from exc
        if not _GENERATION_PATTERN.fullmatch(generation):
            raise ValueError("Vector index CURRENT pointer is invalid.")

        snapshot = root / "snapshots" / generation
        try:
            manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
            chunk_data = json.loads((snapshot / "chunks.json").read_text(encoding="utf-8"))
            index = faiss.read_index(str(snapshot / "index.faiss"))
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            raise ValueError("Saved vector index snapshot is incomplete or unreadable.") from exc

        # v1 缺少身份，v2 只记模型名/路径；二者都必须从文本重新向量化。
        if manifest.get("version") not in {1, 2, _SNAPSHOT_VERSION}:
            raise ValueError("Cannot re-embed an unsupported vector index version.")
        if (
            manifest.get("generation") != generation
            or manifest.get("count") != len(chunk_data)
            or manifest.get("dimension") != index.d
            or index.ntotal != len(chunk_data)
        ):
            raise ValueError("Saved vector index metadata does not match its FAISS index.")

        chunks = [Chunk.from_dict(item) for item in chunk_data]
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Saved vector index contains duplicate chunk IDs.")
        return chunks
