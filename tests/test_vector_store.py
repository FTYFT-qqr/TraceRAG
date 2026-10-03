"""验证 FAISS 快照、模型身份、相似度检索和元数据一致性。"""

import json

import numpy as np
import pytest

from app.config import Settings
from app.models import Chunk, EmbeddingIdentity
from app.vector_store import FaissVectorStore


_TEST_EMBEDDING = EmbeddingIdentity(
    "openai", "test-embedding", endpoint="https://embedding.example/v1"
)


def _chunk(chunk_id: str, index: int = 0) -> Chunk:
    """构造检索测试使用的来源片段。"""
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=f"evidence {chunk_id}",
        file_name="handbook.pdf",
        page_number=index + 1,
        chunk_index=index,
    )


def test_faiss_store_normalizes_vectors_and_round_trips_metadata(tmp_path) -> None:
    """验证 FAISS 索引归一化向量并正确往返来源元数据。"""
    chunks = [_chunk("chunk-a", 0), _chunk("chunk-b", 1)]
    store = FaissVectorStore()
    store.add(chunks, np.array([[3.0, 0.0], [0.0, 2.0]], dtype=np.float32))

    generation = store.save(tmp_path, embedding_identity=_TEST_EMBEDDING)
    restored = FaissVectorStore.load(
        tmp_path, expected_embedding_identity=_TEST_EMBEDDING
    )

    assert len(generation) == 32
    assert restored.count == 2
    assert restored.dimension == 2
    assert restored.chunks == tuple(chunks)
    assert restored.embedding_identity == _TEST_EMBEDDING
    assert list((tmp_path / "snapshots" / generation).iterdir())


def test_faiss_store_rejects_bad_vectors_and_duplicate_chunks() -> None:
    """验证索引拒绝无效向量和重复 Chunk ID。"""
    store = FaissVectorStore()
    chunk = _chunk("chunk-a")

    with pytest.raises(ValueError, match="one two-dimensional"):
        store.add([chunk], np.array([1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="zero vectors"):
        store.add([chunk], np.array([[0.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="unique"):
        store.add([chunk, chunk], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_faiss_store_requires_consistent_dimensions_and_nonempty_save(tmp_path) -> None:
    """验证索引要求向量维度一致且不能保存空索引。"""
    store = FaissVectorStore()
    with pytest.raises(ValueError, match="empty vector store"):
        store.save(tmp_path, embedding_identity=_TEST_EMBEDDING)

    store.add([_chunk("chunk-a")], np.array([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        store.add([_chunk("chunk-b", 1)], np.array([[0.0, 1.0, 0.0]], dtype=np.float32))


def test_faiss_store_detects_snapshot_mismatch(tmp_path) -> None:
    """验证加载索引时检测快照内容不一致。"""
    store = FaissVectorStore()
    store.add([_chunk("chunk-a")], np.array([[1.0, 0.0]], dtype=np.float32))
    generation = store.save(tmp_path, embedding_identity=_TEST_EMBEDDING)
    manifest_path = tmp_path / "snapshots" / generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["count"] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        FaissVectorStore.load(
            tmp_path, expected_embedding_identity=_TEST_EMBEDDING
        )


def test_faiss_store_rejects_same_dimension_different_model(tmp_path) -> None:
    """验证向量维度相同但模型不同的索引不能混用。"""
    store = FaissVectorStore()
    store.add([_chunk("chunk-a")], np.array([[1.0, 0.0]], dtype=np.float32))
    store.save(tmp_path, embedding_identity=_TEST_EMBEDDING)

    with pytest.raises(ValueError, match="does not match"):
        FaissVectorStore.load(
            tmp_path,
            expected_embedding_identity=EmbeddingIdentity(
                "openai", "other-model", endpoint="https://embedding.example/v1"
            ),
        )


def test_faiss_store_rejects_same_alias_from_another_remote_service(tmp_path) -> None:
    """验证同模型别名的不同远程服务端点不能共享索引。"""
    first_settings = Settings(
        embedding_model="shared-alias", base_url="https://service-a.example/v1"
    )
    store = FaissVectorStore()
    store.add([_chunk("chunk-a")], np.array([[1.0, 0.0]], dtype=np.float32))
    store.save(tmp_path, embedding_identity=first_settings.embedding_identity)
    second_settings = Settings(
        embedding_model="shared-alias", base_url="https://service-b.example/v1"
    )

    with pytest.raises(ValueError, match="does not match"):
        FaissVectorStore.load(
            tmp_path, expected_embedding_identity=second_settings.embedding_identity
        )


def test_faiss_store_rejects_replaced_local_weights_at_the_same_path(tmp_path) -> None:
    """验证同一模型路径的替换权重不能沿用旧索引。"""
    original = EmbeddingIdentity(
        "local", "d:/models/bge", model_fingerprint="a" * 64
    )
    replaced = EmbeddingIdentity(
        "local", "d:/models/bge", model_fingerprint="b" * 64
    )
    store = FaissVectorStore()
    store.add([_chunk("chunk-a")], np.array([[1.0, 0.0]], dtype=np.float32))
    store.save(tmp_path, embedding_identity=original)

    with pytest.raises(ValueError, match="does not match"):
        FaissVectorStore.load(tmp_path, expected_embedding_identity=replaced)


def test_faiss_store_loads_when_local_model_fingerprint_is_unchanged(tmp_path) -> None:
    """验证本地模型指纹未变化时可恢复对应索引。"""
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "modules.json").write_text("[]", encoding="utf-8")
    (model_path / "weights.bin").write_bytes(b"stable model content")
    settings = Settings(embedding_provider="local", local_embedding_path=str(model_path))
    store_dir = tmp_path / "index"
    store = FaissVectorStore()
    store.add([_chunk("chunk-a")], np.array([[1.0, 0.0]], dtype=np.float32))
    store.save(store_dir, embedding_identity=settings.embedding_identity)

    restored = FaissVectorStore.load(
        store_dir,
        expected_embedding_identity=Settings(
            embedding_provider="local", local_embedding_path=str(model_path)
        ).embedding_identity,
    )

    assert restored.count == 1
    assert restored.embedding_identity == settings.embedding_identity


def test_faiss_store_rejects_weights_replaced_at_the_same_model_path(tmp_path) -> None:
    """验证同路径替换的模型权重必须重新建索引。"""
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "modules.json").write_text("[]", encoding="utf-8")
    weights_path = model_path / "weights.bin"
    weights_path.write_bytes(b"model revision A")
    first_settings = Settings(
        embedding_provider="local", local_embedding_path=str(model_path)
    )
    store_dir = tmp_path / "index"
    store = FaissVectorStore()
    store.add([_chunk("chunk-a")], np.array([[1.0, 0.0]], dtype=np.float32))
    store.save(store_dir, embedding_identity=first_settings.embedding_identity)

    weights_path.write_bytes(b"model revision B")
    replaced_settings = Settings(
        embedding_provider="local", local_embedding_path=str(model_path)
    )

    assert first_settings.embedding_identity.model == replaced_settings.embedding_identity.model
    assert first_settings.embedding_identity.model_fingerprint != replaced_settings.embedding_identity.model_fingerprint
    with pytest.raises(ValueError, match="does not match"):
        FaissVectorStore.load(
            store_dir, expected_embedding_identity=replaced_settings.embedding_identity
        )


def test_faiss_store_rejects_legacy_snapshot_without_model_identity(tmp_path) -> None:
    """验证缺少模型身份的旧索引不会被当作兼容快照。"""
    store = FaissVectorStore()
    store.add([_chunk("chunk-a")], np.array([[1.0, 0.0]], dtype=np.float32))
    generation = store.save(tmp_path, embedding_identity=_TEST_EMBEDDING)
    manifest_path = tmp_path / "snapshots" / generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = 1
    manifest.pop("embedding_identity")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="no compatible embedding identity"):
        FaissVectorStore.load(
            tmp_path, expected_embedding_identity=_TEST_EMBEDDING
        )


def test_faiss_store_rejects_v2_identity_without_service_endpoint(tmp_path) -> None:
    """验证缺少服务端点的远程旧索引需要重新确认身份。"""
    store = FaissVectorStore()
    store.add([_chunk("chunk-a")], np.array([[1.0, 0.0]], dtype=np.float32))
    generation = store.save(tmp_path, embedding_identity=_TEST_EMBEDDING)
    manifest_path = tmp_path / "snapshots" / generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = 2
    manifest["embedding_identity"] = {"provider": "openai", "model": "test-embedding"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="no compatible embedding identity"):
        FaissVectorStore.load(
            tmp_path, expected_embedding_identity=_TEST_EMBEDDING
        )


def test_faiss_store_can_remove_one_file_without_losing_other_vectors() -> None:
    """验证移除一个文件后其他来源的向量仍然保留。"""
    first = _chunk("chunk-a", 0)
    second = Chunk(
        "chunk-b", "doc-2", "other evidence", "other.txt", None, 0
    )
    store = FaissVectorStore(_TEST_EMBEDDING)
    store.add(
        [first, second],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    candidate = store.without_file_name("HANDBOOK.PDF")

    assert candidate.chunks == (second,)
    assert store.chunks == (first, second)
    assert candidate.search(np.array([0.0, 1.0]), 1)[0][0] == second
