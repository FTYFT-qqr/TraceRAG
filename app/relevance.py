"""基于独立召回通道的原始分数判断是否存在可供回答的候选证据。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.models import RetrievalResult


@dataclass(frozen=True, slots=True)
class RelevancePolicy:
    """分别配置余弦门槛与 BM25 原始分门槛；RRF 仅用于排序。"""

    vector_min_score: float = 0.25
    bm25_min_score: float = 0.0

    def __post_init__(self) -> None:
        """拒绝无穷值及超出含义范围的策略门槛。"""

        if not math.isfinite(self.vector_min_score) or not -1 <= self.vector_min_score <= 1:
            raise ValueError("Vector relevance threshold must be between -1 and 1.")
        if not math.isfinite(self.bm25_min_score) or self.bm25_min_score < 0:
            raise ValueError("BM25 relevance threshold must be finite and nonnegative.")

    def allows(self, results: list[RetrievalResult]) -> bool:
        """任一提供给模型的候选满足对应通道依据时，允许进入回答阶段。"""

        return any(self.supports(result) for result in results)

    def supports(self, result: RetrievalResult) -> bool:
        """按分数类型读取原始相关度，不将融合排序分当作余弦分数。"""

        if result.score_kind == "cosine":
            value = result.vector_score if result.vector_score is not None else result.score
            return math.isfinite(value) and value >= self.vector_min_score
        if result.score_kind == "bm25":
            value = result.bm25_score if result.bm25_score is not None else result.score
            return math.isfinite(value) and value > self.bm25_min_score
        if result.score_kind == "rrf":
            vector = result.vector_score
            sparse = result.bm25_score
            return bool(
                (vector is not None and math.isfinite(vector) and vector >= self.vector_min_score)
                or (sparse is not None and math.isfinite(sparse) and sparse > self.bm25_min_score)
            )
        raise ValueError(f"Unknown retrieval score kind: {result.score_kind}")
