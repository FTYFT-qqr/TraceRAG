"""验证运行时的幂等上传、同名更新和索引重启恢复。"""

import json

import numpy as np
import pytest

from app.config import Settings
from app.models import Chunk
from app.runtime import RAGRuntime
from app.vector_store import FaissVectorStore


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """返回预设向量并记录批量编码调用。"""
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class FakeChatClient:
    def answer(self, query, results) -> str:
        """返回测试预设的回答文本。"""
        return "基于文档回答。[1]"


@pytest.mark.parametrize("strategy,score_kind", [("vector", "cosine"), ("bm25", "bm25"), ("hybrid", "rrf")])
def test_runtime_wires_configured_retrieval_and_refreshes_after_replacement(tmp_path, strategy: str, score_kind: str) -> None:
    """配置召回实际生效，同名替换与重启后两路查询均不返回旧文档。"""

    settings = Settings(api_key="test-key", index_dir=str(tmp_path), retrieval_strategy=strategy)
    runtime = RAGRuntime(settings, FakeEmbedder(), FakeChatClient(), FaissVectorStore())
    runtime.ingest("policy.txt", "旧版政策：可使用旧设备。".encode("utf-8"))
    runtime.ingest("policy.txt", "新版政策：可使用新设备。".encode("utf-8"))
    results = runtime.retriever.retrieve("新设备", top_k=5)
    assert results
    assert all(result.score_kind == score_kind for result in results)
    assert all("旧设备" not in result.chunk.content for result in results)
    restored = RAGRuntime(settings, FakeEmbedder(), FakeChatClient(), FaissVectorStore.load(tmp_path, expected_embedding_identity=settings.embedding_identity))
    assert restored.retriever.retrieve("新设备")[0].score_kind == score_kind
    if strategy == "bm25":
        assert all("旧设备" not in chunk.content for chunk in runtime.retriever.chunks)


def test_online_bm25_uses_its_own_refusal_threshold(tmp_path) -> None:
    """在线稀疏召回使用独立原始分门槛，不套用余弦或融合排序分。"""

    settings = Settings(api_key="test-key", index_dir=str(tmp_path), retrieval_strategy="bm25", bm25_min_score=1000)
    runtime = RAGRuntime(settings, FakeEmbedder(), FakeChatClient(), FaissVectorStore())
    runtime.ingest("policy.txt", "设备可借用。".encode("utf-8"))
    assert runtime.query("设备借用").rejected is True


def test_runtime_ingests_persists_and_deduplicates_document(tmp_path) -> None:
    """验证运行时入库、保存快照并跳过重复文档。"""
    settings = Settings(
        api_key="test-key", embedding_model="test-embedding", index_dir=str(tmp_path)
    )
    runtime = RAGRuntime(settings, FakeEmbedder(), FakeChatClient(), FaissVectorStore())
    content = "年假政策：每年可休十天。".encode("utf-8")

    first = runtime.ingest("policy.txt", content)
    second = runtime.ingest("policy.txt", content)
    other = runtime.ingest("other.txt", "图书馆开放时间为上午九点。".encode("utf-8"))
    updated_content = "新版年假政策：每年可休十五天。".encode("utf-8")
    updated = runtime.ingest("policy.txt", updated_content)
    restored_store = FaissVectorStore.load(
        tmp_path, expected_embedding_identity=settings.embedding_identity
    )
    restored_runtime = RAGRuntime(
        settings, FakeEmbedder(), FakeChatClient(), restored_store
    )
    response = restored_runtime.query("年假有几天？")

    assert first.chunk_count == 1
    assert first.already_indexed is False
    assert second.already_indexed is True
    assert second.generation == first.generation
    assert other.replaced_existing is False
    assert updated.already_indexed is False
    assert updated.replaced_existing is True
    assert runtime.vector_store.count == 2
    assert all(chunk.content != "年假政策：每年可休十天。" for chunk in restored_store.chunks)
    assert any(chunk.content == "新版年假政策：每年可休十五天。" for chunk in restored_store.chunks)
    assert restored_runtime.vector_store.embedding_identity == settings.embedding_identity
    assert response.answer == "基于文档回答。[1]"
    assert response.citations[0].file_name == "policy.txt"
    assert (tmp_path / "CURRENT").is_file()


def test_runtime_rejects_documents_without_extractable_text(tmp_path) -> None:
    """验证运行时拒绝无法提取正文的上传文件。"""
    runtime = RAGRuntime(
        Settings(api_key="test-key", embedding_model="test-embedding", index_dir=str(tmp_path)),
        FakeEmbedder(),
        FakeChatClient(),
        FaissVectorStore(),
    )

    try:
        runtime.ingest("empty.txt", b"\n \t")
    except ValueError as exc:
        assert "No extractable text" in str(exc)
    else:
        raise AssertionError("Expected an empty document to be rejected.")


def test_runtime_rejects_a_loaded_index_from_another_model(tmp_path) -> None:
    """验证运行时拒绝加载其他模型生成的索引。"""
    original_settings = Settings(
        api_key="test-key", embedding_model="model-a", index_dir=str(tmp_path)
    )
    store = FaissVectorStore()
    store.add(
        [
            # 使用真实维度无关的测试向量，专门验证模型身份而非向量数值。
            Chunk("chunk-a", "doc", "evidence", "policy.txt", None, 0)
        ],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    store.save(tmp_path, embedding_identity=original_settings.embedding_identity)
    restored = FaissVectorStore.load(
        tmp_path, expected_embedding_identity=original_settings.embedding_identity
    )
    changed_settings = Settings(
        api_key="test-key", embedding_model="model-b", index_dir=str(tmp_path)
    )

    with pytest.raises(ValueError, match="does not match"):
        RAGRuntime(changed_settings, FakeEmbedder(), FakeChatClient(), restored)




@pytest.mark.parametrize("legacy_version", [1, 2])
def test_runtime_reembeds_legacy_index_and_keeps_latest_same_name_version(
    tmp_path, monkeypatch, legacy_version: int
) -> None:
    """v1 和身份不完整的 v2 快照只取 Chunk 文本重建，旧文件版本不进入新索引。"""
    store = FaissVectorStore()
    legacy_chunks = [
        Chunk("old", "doc-old", "旧版规定", "policy.txt", None, 0),
        Chunk("new", "doc-new", "新版规定", "policy.txt", None, 0),
        Chunk("other", "doc-other", "其他文件", "other.txt", None, 0),
    ]
    store.add(
        legacy_chunks,
        np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )
    old_generation = store.save(
        tmp_path,
        embedding_identity=Settings(embedding_model="old-model").embedding_identity,
    )
    old_manifest_path = tmp_path / "snapshots" / old_generation / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_manifest["version"] = legacy_version
    if legacy_version == 1:
        old_manifest.pop("embedding_identity")
    else:
        old_manifest["embedding_identity"] = {
            "provider": "openai",
            "model": "old-model",
        }
    old_manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")

    monkeypatch.setattr("app.runtime.EmbeddingClient", lambda **_: FakeEmbedder())
    monkeypatch.setattr("app.runtime.ChatClient", lambda **_: FakeChatClient())
    settings = Settings(
        api_key="test-key", embedding_model="current-model", index_dir=str(tmp_path)
    )

    runtime = RAGRuntime.from_settings(settings)
    current_generation = (tmp_path / "CURRENT").read_text(encoding="ascii").strip()
    current_manifest = json.loads(
        (tmp_path / "snapshots" / current_generation / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    restored = FaissVectorStore.load(
        tmp_path, expected_embedding_identity=settings.embedding_identity
    )

    assert current_generation != old_generation
    assert old_manifest_path.is_file()
    assert current_manifest["version"] == 3
    assert current_manifest["embedding_identity"] == settings.embedding_identity.to_dict()
    assert {chunk.chunk_id for chunk in restored.chunks} == {"new", "other"}
    assert runtime.vector_store.search(np.array([1.0, 0.0]), 2)[0][1] == pytest.approx(1.0)
