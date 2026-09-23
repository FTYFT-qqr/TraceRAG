import json

import numpy as np
import pytest

from app.models import Chunk
from app.vector_store import FaissVectorStore


def _chunk(chunk_id: str, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=f"evidence {chunk_id}",
        file_name="handbook.pdf",
        page_number=index + 1,
        chunk_index=index,
    )


def test_faiss_store_normalizes_vectors_and_round_trips_metadata(tmp_path) -> None:
    chunks = [_chunk("chunk-a", 0), _chunk("chunk-b", 1)]
    store = FaissVectorStore()
    store.add(chunks, np.array([[3.0, 0.0], [0.0, 2.0]], dtype=np.float32))

    generation = store.save(tmp_path)
    restored = FaissVectorStore.load(tmp_path)

    assert len(generation) == 32
    assert restored.count == 2
    assert restored.dimension == 2
    assert restored.chunks == tuple(chunks)
    assert list((tmp_path / "snapshots" / generation).iterdir())


def test_faiss_store_rejects_bad_vectors_and_duplicate_chunks() -> None:
    store = FaissVectorStore()
    chunk = _chunk("chunk-a")

    with pytest.raises(ValueError, match="one two-dimensional"):
        store.add([chunk], np.array([1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="zero vectors"):
        store.add([chunk], np.array([[0.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="unique"):
        store.add([chunk, chunk], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_faiss_store_requires_consistent_dimensions_and_nonempty_save(tmp_path) -> None:
    store = FaissVectorStore()
    with pytest.raises(ValueError, match="empty vector store"):
        store.save(tmp_path)

    store.add([_chunk("chunk-a")], np.array([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        store.add([_chunk("chunk-b", 1)], np.array([[0.0, 1.0, 0.0]], dtype=np.float32))


def test_faiss_store_detects_snapshot_mismatch(tmp_path) -> None:
    store = FaissVectorStore()
    store.add([_chunk("chunk-a")], np.array([[1.0, 0.0]], dtype=np.float32))
    generation = store.save(tmp_path)
    manifest_path = tmp_path / "snapshots" / generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["count"] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        FaissVectorStore.load(tmp_path)
