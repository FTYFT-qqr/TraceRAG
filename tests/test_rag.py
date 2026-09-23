import json
from types import SimpleNamespace

import pytest

from app.chat import ChatClient, SYSTEM_PROMPT
from app.models import Chunk, RetrievalResult
from app.rag import REFUSAL_ANSWER, RAGService


def _result(chunk_id: str, *, score: float, page: int = 3) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=f"Evidence text for {chunk_id}.",
        file_name="policy.pdf",
        page_number=page,
        chunk_index=page - 1,
    )
    return RetrievalResult(chunk=chunk, score=score)


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeAnswerGenerator:
    def __init__(self, answer: str) -> None:
        self.response = answer
        self.calls: list[tuple[str, list[RetrievalResult]]] = []

    def answer(self, query: str, results: list[RetrievalResult]) -> str:
        self.calls.append((query, results))
        return self.response


def test_rag_maps_only_used_valid_citations_to_source_chunks() -> None:
    results = [_result("chunk-a", score=0.92), _result("chunk-b", score=0.8, page=4)]
    generator = FakeAnswerGenerator("年假为十天。[1] 其他信息见 [2]，无关标记 [99]。")

    response = RAGService(FakeRetriever(results), generator).query("年假有几天？")

    assert response.rejected is False
    assert "[99]" not in response.answer
    assert [item.chunk_id for item in response.citations] == ["chunk-a", "chunk-b"]
    assert [item.page_number for item in response.citations] == [3, 4]
    assert response.results == tuple(results)


def test_rag_renumbers_sparse_source_ids_to_match_returned_citation_order() -> None:
    results = [_result("first", score=0.95), _result("second", score=0.85, page=4)]
    response = RAGService(
        FakeRetriever(results), FakeAnswerGenerator("只使用第二段证据。[2]")
    ).query("question")

    assert response.answer == "只使用第二段证据。[1]"
    assert [item.chunk_id for item in response.citations] == ["second"]


def test_rag_rejects_low_confidence_without_calling_chat() -> None:
    generator = FakeAnswerGenerator("should not be used [1]")
    retriever = FakeRetriever([_result("low", score=0.24)])

    response = RAGService(retriever, generator, reject_threshold=0.25).query("question")

    assert response.answer == REFUSAL_ANSWER
    assert response.rejected is True
    assert response.citations == ()
    assert generator.calls == []


def test_rag_rejects_empty_retrieval_and_untraceable_answer() -> None:
    empty = RAGService(FakeRetriever([]), FakeAnswerGenerator("answer [1]")).query("question")
    no_citation = RAGService(
        FakeRetriever([_result("evidence", score=1.0)]), FakeAnswerGenerator("answer without source")
    ).query("question")
    unknown_source = RAGService(
        FakeRetriever([_result("evidence", score=1.0)]), FakeAnswerGenerator("answer [8]")
    ).query("question")

    assert empty.rejected and not empty.citations
    assert no_citation.rejected and not no_citation.citations
    assert unknown_source.rejected and not unknown_source.citations


def test_rag_validates_query_top_k_and_threshold() -> None:
    with pytest.raises(ValueError, match="between -1 and 1"):
        RAGService(FakeRetriever([]), FakeAnswerGenerator(""), reject_threshold=1.1)
    service = RAGService(FakeRetriever([]), FakeAnswerGenerator(""))
    with pytest.raises(ValueError, match="Query cannot be empty"):
        service.query("  ")
    with pytest.raises(ValueError, match="top_k"):
        service.query("question", top_k=0)


def test_chat_client_sends_ordered_source_metadata_and_grounding_instructions() -> None:
    class Endpoint:
        request: dict[str, object] | None = None

        def create(self, **kwargs: object) -> SimpleNamespace:
            self.request = kwargs
            message = SimpleNamespace(content="年假十天。[1]")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    endpoint = Endpoint()
    client = ChatClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-chat",
        client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint)),
    )
    results = [_result("first", score=0.9), _result("second", score=0.8, page=7)]

    answer = client.answer("年假几天？", results)

    request = endpoint.request
    assert answer == "年假十天。[1]"
    assert request is not None
    assert request["model"] == "test-chat"
    assert request["temperature"] == 0
    messages = request["messages"]
    assert "untrusted quoted data" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["question"] == "年假几天？"
    assert [source["chunk_id"] for source in payload["evidence"]] == ["first", "second"]
    assert [source["id"] for source in payload["evidence"]] == [1, 2]
    assert SYSTEM_PROMPT == messages[0]["content"]


def test_chat_client_requires_credentials_and_nonempty_output() -> None:
    with pytest.raises(ValueError, match="TRACERAG_API_KEY"):
        ChatClient(api_key=None, base_url=None, model="chat")

    class EmptyEndpoint:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
            )

    client = ChatClient(
        api_key="test-key",
        base_url=None,
        model="chat",
        client=SimpleNamespace(chat=SimpleNamespace(completions=EmptyEndpoint())),
    )
    with pytest.raises(ValueError, match="empty answer"):
        client.answer("question", [_result("source", score=1.0)])
