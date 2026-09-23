"""Evidence-grounded RAG answer generation, citations, and refusal policy."""

from __future__ import annotations

import re
from typing import Protocol

from app.models import Citation, QueryResponse, RetrievalResult


REFUSAL_ANSWER = "根据当前知识库，我无法确认这个问题。"
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]: ...


class AnswerGenerator(Protocol):
    def answer(self, query: str, results: list[RetrievalResult]) -> str: ...


class RAGService:
    """Retrieve evidence, reject low-confidence questions, and cite used chunks."""

    def __init__(
        self,
        retriever: Retriever,
        chat_client: AnswerGenerator,
        *,
        reject_threshold: float = 0.25,
    ) -> None:
        if not -1.0 <= reject_threshold <= 1.0:
            raise ValueError("reject_threshold must be between -1 and 1.")
        self._retriever = retriever
        self._chat_client = chat_client
        self.reject_threshold = reject_threshold

    def query(self, query: str, top_k: int = 5) -> QueryResponse:
        """Answer a question only when retrieval passes the relevance threshold."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        results = self._retriever.retrieve(query, top_k=top_k)
        if not results or results[0].score < self.reject_threshold:
            return QueryResponse(REFUSAL_ANSWER, True, (), tuple(results))

        generated = self._chat_client.answer(query, results)
        source_ids = {index for index, _ in enumerate(results, start=1)}
        used_ids: list[int] = []
        for match in _CITATION_PATTERN.finditer(generated):
            source_id = int(match.group(1))
            if source_id in source_ids and source_id not in used_ids:
                used_ids.append(source_id)

        if not used_ids:
            return QueryResponse(REFUSAL_ANSWER, True, (), tuple(results))

        citation_labels = {
            source_id: index for index, source_id in enumerate(used_ids, start=1)
        }
        answer = _CITATION_PATTERN.sub(
            lambda match: f"[{citation_labels[int(match.group(1))]}]"
            if int(match.group(1)) in citation_labels
            else "",
            generated,
        ).strip()
        if not answer:
            return QueryResponse(REFUSAL_ANSWER, True, (), tuple(results))

        citations = tuple(
            Citation(
                chunk_id=results[source_id - 1].chunk.chunk_id,
                file_name=results[source_id - 1].chunk.file_name,
                page_number=results[source_id - 1].chunk.page_number,
                text=results[source_id - 1].chunk.content,
            )
            for source_id in used_ids
        )
        return QueryResponse(answer, False, citations, tuple(results))
