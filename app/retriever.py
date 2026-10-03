"""将查询编码为向量，并从 FAISS 返回带来源信息的 Top-K 片段。"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from app.embeddings import EmbeddingClient
from app.models import RetrievalResult
from app.vector_store import FaissVectorStore


class TextEmbedder(Protocol):
    """统一远程和本地文本向量化客户端的调用形式。"""

    def embed_texts(self, texts: list[str]):
        """将输入文本按原顺序编码为二维向量矩阵。"""
        ...


class VectorRetriever:
    """封装查询向量化和 FAISS 检索，保持二者的模型配置一致。"""

    def __init__(self, embedder: EmbeddingClient | TextEmbedder, store: FaissVectorStore) -> None:
        """注入向量化客户端和索引，保持检索层与模型解耦。"""
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """拒绝空问题或无效 Top-K，再返回按余弦分数排序的片段。"""
        return self.retrieve_many([query], top_k=top_k)[0]

    def retrieve_many(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[RetrievalResult]]:
        """批量编码多个查询以减少本地模型重复调用，并保持输入顺序。"""

        if any(not query.strip() for query in queries):
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not queries:
            return []
        if self._store.count == 0:
            return [[] for _ in queries]

        vectors = self._embedder.embed_texts(queries)
        if vectors.ndim != 2 or vectors.shape[0] != len(queries):
            raise ValueError("Query embedding count does not match the query batch.")
        all_results = []
        for vector in vectors:
            matches = self._store.search(vector, top_k)
            all_results.append(
                [
                    RetrievalResult(chunk=chunk, score=score, candidate_rank=rank, vector_score=score)
                    for rank, (chunk, score) in enumerate(matches, start=1)
                ]
            )
        return all_results


# 中英文混合问题使用英文词、数字和中文字符三类 token；中文按字符切分，
# 可以在没有额外分词依赖时稳定处理专有名词和短句。
_SPARSE_TOKEN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


def tokenize_for_bm25(text: str) -> list[str]:
    """将文本转换为 BM25 使用的轻量 token 序列。"""

    tokens = []
    for token in _SPARSE_TOKEN.findall(text):
        normalized = token.casefold()
        if re.fullmatch(r"[\u4e00-\u9fff]+", normalized):
            # 中文不依赖外部分词器，以相邻二字词降低单字噪声。
            tokens.extend(
                normalized[index : index + 2]
                for index in range(max(1, len(normalized) - 1))
            )
        else:
            tokens.append(normalized)
    return tokens


class BM25Retriever:
    """基于内存倒排统计的 BM25 检索器，不依赖第三方稀疏检索包。"""

    def __init__(
        self,
        chunks: list,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """构建 BM25 的文档词频、文档频率和长度统计。"""
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 k1 must be positive and b must be between 0 and 1.")
        self._chunks = tuple(chunks)
        self.k1 = k1
        self.b = b
        self._tokenized = [tokenize_for_bm25(chunk.content) for chunk in self._chunks]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokenized]
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokenized:
            document_frequency.update(set(tokens))
        self._document_frequency = document_frequency
        self._document_count = len(self._chunks)
        self._average_length = (
            sum(len(tokens) for tokens in self._tokenized) / self._document_count
            if self._document_count
            else 0.0
        )

    @property
    def chunks(self) -> tuple:
        """返回构建倒排表时使用的有序片段。"""

        return self._chunks

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """按 BM25 分数返回候选，并记录稀疏检索中的原始名次。"""

        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not self._chunks:
            return []

        query_terms = Counter(tokenize_for_bm25(query))
        scores: list[tuple[int, float]] = []
        for index, frequencies in enumerate(self._term_frequencies):
            document_length = len(self._tokenized[index])
            score = 0.0
            for term, query_frequency in query_terms.items():
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                # BM25 的 IDF 使用 +1 平滑，避免小语料库中出现负分。
                idf = math.log(
                    1.0
                    + (self._document_count - self._document_frequency.get(term, 0) + 0.5)
                    / (self._document_frequency.get(term, 0) + 0.5)
                )
                length_ratio = (
                    document_length / self._average_length
                    if self._average_length
                    else 1.0
                )
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += idf * (term_frequency * (self.k1 + 1) / denominator) * query_frequency
            # 零词面命中的片段不参与排序或融合，避免无关文档获得 RRF 贡献。
            if score > 0:
                scores.append((index, score))

        # 分数相同时保留文档原始顺序，保证报告可复现。
        scores.sort(key=lambda item: (-item[1], item[0]))
        return [
            RetrievalResult(
                chunk=self._chunks[index],
                score=float(score),
                candidate_rank=rank,
                score_kind="bm25",
                bm25_score=float(score),
            )
            for rank, (index, score) in enumerate(scores[:top_k], start=1)
        ]

    def retrieve_many(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[RetrievalResult]]:
        """保持查询列表顺序批量计算 BM25 结果，便于评测重复使用。"""

        return [self.retrieve(query, top_k=top_k) for query in queries]


class HybridRetriever:
    """用 Reciprocal Rank Fusion 合并向量和 BM25 两路召回。"""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        sparse_retriever: BM25Retriever,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
    ) -> None:
        """注入两路检索器和 RRF 参数，避免混合逻辑依赖具体向量实现。"""
        if rrf_k <= 0 or candidate_multiplier <= 0:
            raise ValueError("RRF parameters must be positive integers.")
        self._vector = vector_retriever
        self._sparse = sparse_retriever
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """用两路原始排名计算 RRF，同时保留各通道的相关度。"""

        return self.retrieve_many([query], top_k=top_k)[0]

    def retrieve_many(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[RetrievalResult]]:
        """分别批量查询两路检索，并按 RRF 分数合并每道问题的候选。"""

        if any(not query.strip() for query in queries):
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not queries:
            return []
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        vector_results = self._vector.retrieve_many(queries, candidate_k)
        sparse_results = self._sparse.retrieve_many(queries, candidate_k)
        if len(vector_results) != len(queries) or len(sparse_results) != len(queries):
            raise ValueError("Hybrid retrievers must return one ranking per query.")
        return [
            self._fuse_results(vector, sparse, top_k)
            for vector, sparse in zip(vector_results, sparse_results)
        ]

    def _fuse_results(
        self,
        vector_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """合并两路排名并保留原始分数；融合值只用于候选排序。"""

        merged: dict[str, tuple[RetrievalResult, float]] = {}
        vector_scores = {result.chunk.chunk_id: result.score for result in vector_results}
        sparse_scores = {result.chunk.chunk_id: result.score for result in sparse_results if result.score > 0}
        for results in (vector_results, [result for result in sparse_results if result.score > 0]):
            for rank, result in enumerate(results, start=1):
                contribution = 1.0 / (self.rrf_k + rank)
                previous = merged.get(result.chunk.chunk_id)
                if previous is None:
                    merged[result.chunk.chunk_id] = (result, contribution)
                else:
                    merged[result.chunk.chunk_id] = (previous[0], previous[1] + contribution)

        ranked = sorted(
            merged.values(),
            key=lambda item: (-item[1], item[0].chunk.chunk_id),
        )[:top_k]
        if not ranked:
            return []
        return [
            RetrievalResult(
                chunk=result.chunk,
                score=float(score),
                candidate_rank=rank,
                score_kind="rrf",
                vector_score=vector_scores.get(result.chunk.chunk_id),
                bm25_score=sparse_scores.get(result.chunk.chunk_id),
            )
            for rank, (result, score) in enumerate(ranked, start=1)
        ]
