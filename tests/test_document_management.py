"""覆盖 007 要求的目录、删除、检索清理和快照回退行为。"""

from __future__ import annotations

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.runtime as runtime_module
from app.config import Settings
from app.main import create_app
from app.models import DocumentPage
from app.runtime import RAGRuntime
from app.vector_store import FaissVectorStore


class _FakeEmbedder:
    """用固定维度向量隔离外部模型，保留真实 FAISS 读写链路。"""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """为每段文本返回有效的二维非零向量。"""
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class _FakeChatClient:
    """避免文档管理回归测试发出真实 Chat 请求。"""

    def answer(self, query, results) -> str:
        """为有证据的查询返回可引用的固定回答。"""
        return "已找到资料。[1]"


def _runtime(tmp_path, strategy: str = "hybrid") -> RAGRuntime:
    """使用隔离目录创建带真实检索器和快照的运行时。"""
    settings = Settings(
        api_key="test-key",
        index_dir=str(tmp_path),
        retrieval_strategy=strategy,
    )
    return RAGRuntime(settings, _FakeEmbedder(), _FakeChatClient(), FaissVectorStore())


def _restore(runtime: RAGRuntime, monkeypatch) -> RAGRuntime:
    """从磁盘 CURRENT 重新创建运行时，模拟应用进程重启。"""
    monkeypatch.setattr("app.runtime.EmbeddingClient", lambda **_: _FakeEmbedder())
    monkeypatch.setattr("app.runtime.ChatClient", lambda **_: _FakeChatClient())
    return RAGRuntime.from_settings(runtime.settings)


def test_document_api_lists_real_counts_and_indexed_pdf_pages(tmp_path, monkeypatch) -> None:
    """目录只计入实际片段及有文本的 PDF 页，删除响应与目录一致。"""
    runtime = _runtime(tmp_path)
    original_loader = runtime_module.load_document

    def load_test_document(file_name: str, data: bytes):
        """仅替换 PDF 解析输入，使页码边界不依赖 PDF 生成库。"""
        if file_name == "report.pdf":
            return [
                DocumentPage("pdf-id", file_name, 1, "第一页设备检查记录。"),
                DocumentPage("pdf-id", file_name, 2, "  \n"),
                DocumentPage("pdf-id", file_name, 3, "第三页维修要求。" * 100),
            ]
        return original_loader(file_name, data)

    monkeypatch.setattr("app.runtime.load_document", load_test_document)
    with TestClient(create_app(runtime.settings, runtime_factory=lambda _: runtime)) as client:
        empty = client.get("/documents")
        text_upload = client.post("/documents", files={"file": ("notice.txt", "公告：设备可借用。".encode())})
        pdf_upload = client.post("/documents", files={"file": ("report.pdf", b"pdf-test-input")})
        listing = client.get("/documents")

        assert empty.status_code == 200
        assert empty.json()["documents"] == []
        assert empty.json()["index_generation"] == ""
        assert text_upload.status_code == pdf_upload.status_code == listing.status_code == 200
        documents = {item["file_name"]: item for item in listing.json()["documents"]}
        assert listing.json()["document_count"] == 2
        assert listing.json()["chunk_count"] == sum(item["chunk_count"] for item in documents.values())
        assert documents["notice.txt"]["chunk_count"] == 1
        assert documents["notice.txt"]["indexed_page_count"] is None
        assert documents["report.pdf"]["chunk_count"] == 3
        assert documents["report.pdf"]["indexed_page_count"] == 2
        assert listing.json()["index_generation"] == pdf_upload.json()["index_generation"]

        unknown = client.delete("/documents/no-such-id")
        assert unknown.status_code == 404
        assert client.get("/documents").json()["index_generation"] == listing.json()["index_generation"]

        removed = client.delete(f"/documents/{documents['report.pdf']['document_id']}")
        assert removed.status_code == 200
        assert removed.json()["deleted_chunk_count"] == 3
        assert removed.json()["document_count"] == 1
        assert removed.json()["chunk_count"] == 1
        assert client.get("/documents").json()["documents"][0]["file_name"] == "notice.txt"


@pytest.mark.parametrize("strategy", ["vector", "bm25", "hybrid"])
def test_delete_clears_every_retrieval_strategy_and_survives_restart(tmp_path, monkeypatch, strategy) -> None:
    """删除后原检索器、重新加载的检索器及目录均不得暴露旧片段。"""
    runtime = _runtime(tmp_path, strategy)
    removed_doc = runtime.ingest("old.txt", "旧版唯一条款：红色通行证。".encode())
    retained_doc = runtime.ingest("keep.txt", "保留条款：蓝色通行证。".encode())
    assert runtime.retriever.retrieve("红色通行证", top_k=5)

    deletion = runtime.delete_document(removed_doc.document_id)
    assert deletion.file_name == "old.txt"
    assert deletion.document_count == 1
    assert deletion.chunk_count == 1
    assert deletion.generation != retained_doc.generation
    assert {document.document_id for document in runtime.list_documents()} == {retained_doc.document_id}
    assert all(result.chunk.document_id != removed_doc.document_id for result in runtime.retriever.retrieve("红色通行证", top_k=5))

    restarted = _restore(runtime, monkeypatch)
    assert {document.document_id for document in restarted.list_documents()} == {retained_doc.document_id}
    assert all(result.chunk.document_id != removed_doc.document_id for result in restarted.retriever.retrieve("红色通行证", top_k=5))
    assert restarted.vector_store.count == 1
    if strategy == "bm25":
        assert {chunk.document_id for chunk in restarted.retriever.chunks} == {retained_doc.document_id}


@pytest.mark.parametrize("strategy", ["vector", "bm25", "hybrid"])
def test_last_document_delete_persists_empty_snapshot_and_allows_reupload(tmp_path, monkeypatch, strategy) -> None:
    """删除最后一份资料后仍保存空索引，并允许重启后再次上传。"""
    runtime = _runtime(tmp_path, strategy)
    first = runtime.ingest("single.txt", "唯一资料：黄色凭证。".encode())
    deletion = runtime.delete_document(first.document_id)
    assert deletion.document_count == deletion.chunk_count == 0
    assert deletion.generation != first.generation
    assert (tmp_path / "CURRENT").read_text(encoding="ascii").strip() == deletion.generation
    assert runtime.retriever.retrieve("黄色凭证") == []
    assert runtime.query("黄色凭证").rejected is True

    restarted = _restore(runtime, monkeypatch)
    assert restarted.list_documents() == []
    assert restarted.vector_store.count == 0
    assert restarted.vector_store.dimension == 2
    assert restarted.retriever.retrieve("黄色凭证") == []
    again = restarted.ingest("fresh.txt", "新增资料：绿色凭证。".encode())
    assert again.generation != deletion.generation
    assert again.chunk_count == 1
    assert [document.file_name for document in restarted.list_documents()] == ["fresh.txt"]


def test_failed_snapshot_pointer_commit_keeps_generation_and_retrieval(tmp_path, monkeypatch) -> None:
    """提交 CURRENT 失败时，磁盘和内存继续提供旧代检索结果。"""
    runtime = _runtime(tmp_path)
    old = runtime.ingest("old.txt", "旧资料：请使用红色通行证。".encode())
    current = (tmp_path / "CURRENT").read_text(encoding="ascii").strip()
    old_store = runtime.vector_store
    old_retriever = runtime.retriever
    old_rag = runtime.rag
    replace = os.replace

    def fail_pointer_commit(source, destination):
        """只阻止 CURRENT 提交，允许候选快照先写完。"""
        if str(destination) == str(tmp_path / "CURRENT"):
            raise OSError("simulated CURRENT commit failure")
        return replace(source, destination)

    monkeypatch.setattr("app.vector_store.os.replace", fail_pointer_commit)
    with TestClient(create_app(runtime.settings, runtime_factory=lambda _: runtime)) as client:
        failed = client.delete(f"/documents/{old.document_id}")
        assert failed.status_code == 503
        assert "CURRENT commit failure" not in failed.text
        assert client.get("/documents").json()["index_generation"] == current
    assert runtime.vector_store is old_store
    assert runtime.retriever is old_retriever
    assert runtime.rag is old_rag
    assert runtime._current_generation() == current
    assert {document.document_id for document in runtime.list_documents()} == {old.document_id}
    assert runtime.retriever.retrieve("红色通行证")[0].chunk.document_id == old.document_id

    monkeypatch.setattr("app.vector_store.os.replace", replace)
    restarted = _restore(runtime, monkeypatch)
    assert restarted._current_generation() == current
    assert [document.document_id for document in restarted.list_documents()] == [old.document_id]
    deletion = restarted.delete_document(old.document_id)
    assert deletion.generation != current


def test_failed_upload_pointer_commit_keeps_existing_generation(tmp_path, monkeypatch) -> None:
    """上传新文档提交失败时不切换索引，重试可正常追加。"""
    runtime = _runtime(tmp_path)
    first = runtime.ingest("first.txt", "第一份资料：蓝色凭证。".encode())
    replace = os.replace

    def fail_pointer_commit(source, destination):
        """只模拟新上传在 CURRENT 指针切换处失败。"""
        if str(destination) == str(tmp_path / "CURRENT"):
            raise OSError("simulated upload commit failure")
        return replace(source, destination)

    monkeypatch.setattr("app.vector_store.os.replace", fail_pointer_commit)
    with pytest.raises(OSError, match="upload commit failure"):
        runtime.ingest("second.txt", "第二份资料：绿色凭证。".encode())
    assert runtime._current_generation() == first.generation
    assert [document.file_name for document in runtime.list_documents()] == ["first.txt"]
    assert {chunk.file_name for chunk in runtime.vector_store.chunks} == {"first.txt"}

    monkeypatch.setattr("app.vector_store.os.replace", replace)
    again = runtime.ingest("second.txt", "第二份资料：绿色凭证。".encode())
    assert again.generation != first.generation
    assert [document.file_name for document in runtime.list_documents()] == ["first.txt", "second.txt"]
