"""验证 BM25 与混合召回使用原始相关度拒答，融合排名不决定证据充分性。"""

import pytest

from app.models import Chunk, RetrievalResult
from app.rag import RAGService
from app.relevance import RelevancePolicy
from app.retriever import BM25Retriever, HybridRetriever


class FixedVector:
    """以固定原始余弦分隔离验证混合召回门槛。"""

    def __init__(self, chunk: Chunk, score: float) -> None:
        """保存片段和测试要求的相关度。"""
        self.chunk, self.score = chunk, score

    def retrieve_many(self, queries: list[str], top_k: int = 5) -> list[list[RetrievalResult]]:
        """返回与问题顺序一致的人工向量排名。"""
        return [[RetrievalResult(self.chunk, self.score, 1, vector_score=self.score)] for _ in queries]


class RecordingGenerator:
    """通过调用计数验证无证据时不会请求回答模型。"""

    def __init__(self) -> None:
        """创建本地调用计数。"""
        self.calls = 0

    def answer(self, query: str, results: list[RetrievalResult]) -> str:
        """用当前证据构造带有效编号的回答。"""
        self.calls += 1
        return f"{results[0].chunk.content}[1]"


def _chunk() -> Chunk:
    """建立门槛测试所需的固定图书馆证据。"""
    return Chunk("library", "doc", "图书馆开放时间", "audit.txt", None, 0)


def test_bm25_removes_zero_overlap_candidates() -> None:
    """完全无词面命中时 BM25 返回空候选并在模型调用前拒答。"""
    retriever = BM25Retriever([_chunk()])
    generator = RecordingGenerator()
    assert retriever.retrieve("火星矿场保险价格") == []
    response = RAGService(retriever, generator).query("火星矿场保险价格")
    assert response.rejected is True
    assert generator.calls == 0


def test_hybrid_rejects_when_both_original_channels_lack_support() -> None:
    """文档中的零向量分且零词面命中不得因融合排名产生虚假置信度。"""
    chunk = _chunk()
    retriever = HybridRetriever(FixedVector(chunk, 0.0), BM25Retriever([chunk]))
    results = retriever.retrieve("火星矿场保险价格")
    assert results[0].score > 0
    assert results[0].score_kind == "rrf"
    assert results[0].vector_score == 0.0
    assert results[0].bm25_score is None
    generator = RecordingGenerator()
    response = RAGService(retriever, generator).query("火星矿场保险价格")
    assert response.rejected is True
    assert generator.calls == 0


@pytest.mark.parametrize("query,vector_score", [("图书馆开放时间", 0.0), ("何时可进入阅览区域", 0.9)])
def test_hybrid_answers_with_sparse_or_vector_support(query: str, vector_score: float) -> None:
    """任一原始通道存在达标证据时，混合策略允许正常回答。"""
    chunk = _chunk()
    retriever = HybridRetriever(FixedVector(chunk, vector_score), BM25Retriever([chunk]))
    generator = RecordingGenerator()
    response = RAGService(retriever, generator).query(query)
    assert response.rejected is False
    assert generator.calls == 1


def test_bm25_threshold_is_independent_of_cosine_threshold() -> None:
    """BM25 正词面命中使用自己的原始分门槛，调整余弦门槛不改变其判定。"""
    results = BM25Retriever([_chunk()]).retrieve("图书馆开放时间")
    assert RelevancePolicy(vector_min_score=1.0).allows(results) is True
    assert RelevancePolicy(bm25_min_score=results[0].score).allows(results) is False


def test_rrf_rank_value_alone_cannot_allow_an_answer() -> None:
    """即使人为提高融合排序分，缺少原始通道支持仍须拒答。"""
    result = RetrievalResult(_chunk(), 999, 1, "rrf", 0.0, 0.0)
    assert RelevancePolicy().allows([result]) is False


@pytest.mark.parametrize("kwargs", [{"vector_min_score": float("nan")}, {"bm25_min_score": -1}, {"bm25_min_score": float("inf")}])
def test_relevance_policy_rejects_invalid_thresholds(kwargs: dict) -> None:
    """无效门槛不能悄悄造成全部放行或全部拒答。"""
    with pytest.raises(ValueError):
        RelevancePolicy(**kwargs)
