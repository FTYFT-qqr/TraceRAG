"""验证 Top-K 排序、来源信息和查询向量维度检查。"""

import numpy as np
import pytest

from app.models import Chunk
from app.retriever import BM25Retriever, HybridRetriever, VectorRetriever
from app.vector_store import FaissVectorStore


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        """初始化测试替身及其记录状态。"""
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """返回预设向量并记录批量编码调用。"""
        self.calls.append(texts)
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def _chunk(chunk_id: str, content: str, index: int) -> Chunk:
    """构造检索测试使用的来源片段。"""
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=content,
        file_name="handbook.pdf",
        page_number=index + 1,
        chunk_index=index,
    )


def test_retriever_returns_ranked_chunks_scores_and_source_metadata() -> None:
    """验证向量检索按分数排序并保留片段来源元数据。"""
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
    """验证检索结果数量不超过索引规模且空索引安全返回。"""
    store = FaissVectorStore()
    store.add([_chunk("only", "single evidence", 0)], np.array([[1, 0]], dtype=np.float32))
    embedder = FakeEmbedder({"query": [1, 0]})

    assert len(VectorRetriever(embedder, store).retrieve("query", top_k=10)) == 1
    assert VectorRetriever(FakeEmbedder({}), FaissVectorStore()).retrieve("query") == []


def test_retriever_validates_query_top_k_and_embedding_shape() -> None:
    """验证检索器校验问题、Top-K 和查询向量维度。"""
    store = FaissVectorStore()
    store.add([_chunk("only", "evidence", 0)], np.array([[1, 0]], dtype=np.float32))
    retriever = VectorRetriever(FakeEmbedder({"query": [1, 0], "wrong": [1, 0, 0]}), store)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve("  ")
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("query", top_k=0)
    with pytest.raises(ValueError, match="dimension"):
        retriever.retrieve("wrong")


def test_retriever_batches_query_embeddings_in_input_order() -> None:
    """确认批量查询只调用一次向量模型并保持问题顺序。"""

    chunks = [_chunk("first", "first evidence", 0), _chunk("second", "second evidence", 1)]
    store = FaissVectorStore()
    store.add(chunks, np.array([[1, 0], [0, 1]], dtype=np.float32))
    embedder = FakeEmbedder({"q1": [1, 0], "q2": [0, 1]})

    results = VectorRetriever(embedder, store).retrieve_many(["q1", "q2"], top_k=1)

    assert [items[0].chunk.chunk_id for items in results] == ["first", "second"]
    assert embedder.calls == [["q1", "q2"]]


def test_retrieval_hits_expected_evidence_for_twelve_queries() -> None:
    """用正交人工向量隔离验证排序逻辑，不代表真实模型召回率。"""
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


def test_bm25_retriever_returns_keyword_match_with_stable_rank() -> None:
    """确认中文关键词 BM25 检索能定位正确片段且排名稳定。"""

    chunks = [
        _chunk("address", "图书馆位于松林路18号", 0),
        _chunk("hours", "图书馆周末上午9点开放", 1),
    ]

    results = BM25Retriever(chunks).retrieve("松林路18号", top_k=2)

    assert results[0].chunk.chunk_id == "address"
    assert results[0].candidate_rank == 1
    assert results[0].score > 0


def test_hybrid_retriever_merges_vector_and_sparse_rankings() -> None:
    """确认 RRF 能融合向量与关键词两路顺序并保留名次。"""

    chunks = [
        _chunk("address", "松林路18号", 0),
        _chunk("hours", "周末上午9点开放", 1),
    ]
    store = FaissVectorStore()
    store.add(chunks, np.array([[1, 0], [0, 1]], dtype=np.float32))
    vector = VectorRetriever(FakeEmbedder({"松林路": [0, 1]}), store)
    hybrid = HybridRetriever(vector, BM25Retriever(chunks), rrf_k=10)

    results = hybrid.retrieve("松林路", top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["address", "hours"]
    assert [result.candidate_rank for result in results] == [1, 2]
    assert results[0].score == pytest.approx(1 / 12 + 1 / 11)
    assert results[0].score_kind == "rrf"
    assert results[0].vector_score == pytest.approx(0.0)
    assert results[0].bm25_score > 0
