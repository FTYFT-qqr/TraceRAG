import numpy as np
import pytest

from app.models import Chunk
from app.retriever import VectorRetriever
from app.vector_store import FaissVectorStore


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def _chunk(chunk_id: str, content: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=content,
        file_name="handbook.pdf",
        page_number=index + 1,
        chunk_index=index,
    )


def test_retriever_returns_ranked_chunks_scores_and_source_metadata() -> None:
    chunks = [
        _chunk("policy", "annual leave is ten days", 0),
        _chunk("holiday", "office holiday calendar", 1),
        _chunk("mixed", "leave and holiday policy", 2),
    ]
    store = FaissVectorStore()
    store.add(chunks, np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    embedder = FakeEmbedder({"annual leave": [1, 0]})

    results = VectorRetriever(embedder, store).retrieve("annual leave", top_k=2)

    assert [item.chunk.chunk_id for item in results] == ["policy", "mixed"]
    assert [item.score for item in results] == pytest.approx([1.0, 2**-0.5])
    assert results[0].chunk.content == "annual leave is ten days"
    assert results[0].chunk.file_name == "handbook.pdf"
    assert results[0].chunk.page_number == 1
    assert embedder.calls == [["annual leave"]]


def test_retriever_caps_top_k_to_index_size_and_handles_empty_store() -> None:
    store = FaissVectorStore()
    store.add([_chunk("only", "single evidence", 0)], np.array([[1, 0]], dtype=np.float32))
    embedder = FakeEmbedder({"query": [1, 0]})

    assert len(VectorRetriever(embedder, store).retrieve("query", top_k=10)) == 1
    assert VectorRetriever(FakeEmbedder({}), FaissVectorStore()).retrieve("query") == []


def test_retriever_validates_query_top_k_and_embedding_shape() -> None:
    store = FaissVectorStore()
    store.add([_chunk("only", "evidence", 0)], np.array([[1, 0]], dtype=np.float32))
    retriever = VectorRetriever(FakeEmbedder({"query": [1, 0], "wrong": [1, 0, 0]}), store)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve("  ")
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("query", top_k=0)
    with pytest.raises(ValueError, match="dimension"):
        retriever.retrieve("wrong")


def test_retrieval_hits_expected_evidence_for_twelve_queries() -> None:
    dimension = 12
    chunks = [
        _chunk(f"chunk-{index}", f"evidence for query {index}", index)
        for index in range(dimension)
    ]
    vectors = np.eye(dimension, dtype=np.float32)
    store = FaissVectorStore()
    store.add(chunks, vectors)
    queries = {f"query-{index}": vectors[index].tolist() for index in range(dimension)}
    retriever = VectorRetriever(FakeEmbedder(queries), store)

    for index in range(dimension):
        results = retriever.retrieve(f"query-{index}", top_k=3)
        assert results[0].chunk.chunk_id == f"chunk-{index}"
        assert results[0].score == pytest.approx(1.0)
