"""FAISS vector storage with a JSON mapping to source-aware chunks."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

import faiss
import numpy as np

from app.models import Chunk


_SNAPSHOT_VERSION = 1
_GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class FaissVectorStore:
    """Normalized inner-product FAISS index aligned with an ordered chunk list."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    @property
    def count(self) -> int:
        return len(self._chunks)

    @property
    def dimension(self) -> int | None:
        return self._index.d if self._index is not None else None

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)


    def search(self, embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        """Return the top cosine matches with their source metadata."""

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
        """Add aligned chunk/vector rows, normalizing them for cosine search."""

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

    def save(self, directory: str | Path) -> str:
        """Write an immutable snapshot and atomically switch its CURRENT pointer."""

        if self._index is None or not self._chunks:
            raise ValueError("Cannot save an empty vector store.")

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
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, destination)
            pointer_tmp = root / f".CURRENT-{generation}.tmp"
            pointer_tmp.write_text(generation, encoding="ascii")
            os.replace(pointer_tmp, root / "CURRENT")
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return generation

    @classmethod
    def load(cls, directory: str | Path) -> "FaissVectorStore":
        """Load and validate the immutable snapshot named by CURRENT."""

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

        if (
            manifest.get("version") != _SNAPSHOT_VERSION
            or manifest.get("generation") != generation
            or manifest.get("count") != len(chunk_data)
            or manifest.get("dimension") != index.d
            or index.ntotal != len(chunk_data)
        ):
            raise ValueError("Saved vector index metadata does not match its FAISS index.")

        store = cls()
        store._index = index
        store._chunks = [Chunk.from_dict(item) for item in chunk_data]
        if len({chunk.chunk_id for chunk in store._chunks}) != len(store._chunks):
            raise ValueError("Saved vector index contains duplicate chunk IDs.")
        return store
