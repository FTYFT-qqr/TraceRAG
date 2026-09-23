import numpy as np

from app.config import Settings
from app.runtime import RAGRuntime
from app.vector_store import FaissVectorStore


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class FakeChatClient:
    def answer(self, query, results) -> str:
        return "基于文档回答。[1]"


def test_runtime_ingests_persists_and_deduplicates_document(tmp_path) -> None:
    settings = Settings(api_key="test-key", index_dir=str(tmp_path))
    runtime = RAGRuntime(settings, FakeEmbedder(), FakeChatClient(), FaissVectorStore())
    content = "年假政策：每年可休十天。".encode("utf-8")

    first = runtime.ingest("policy.txt", content)
    second = runtime.ingest("policy.txt", content)
    response = runtime.query("年假有几天？")

    assert first.chunk_count == 1
    assert first.already_indexed is False
    assert second.already_indexed is True
    assert second.generation == first.generation
    assert runtime.vector_store.count == 1
    assert response.answer == "基于文档回答。[1]"
    assert response.citations[0].file_name == "policy.txt"
    assert (tmp_path / "CURRENT").is_file()


def test_runtime_rejects_documents_without_extractable_text(tmp_path) -> None:
    runtime = RAGRuntime(
        Settings(api_key="test-key", index_dir=str(tmp_path)),
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
