"""验证拒答优先级、引用编号校验及 Chat 请求中的证据顺序。"""

import json
from types import SimpleNamespace

import pytest

from app.chat import ChatClient, SYSTEM_PROMPT
from app.models import Chunk, RetrievalResult
from app.rag import REFUSAL_ANSWER, RAGService


def _result(chunk_id: str, *, score: float, page: int = 3) -> RetrievalResult:
    """构造 RAG 测试所需的带来源检索结果。"""
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
        """初始化测试替身及其记录状态。"""
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """返回固定候选，用于隔离验证 RAG 逻辑。"""
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeAnswerGenerator:
    def __init__(self, answer: str) -> None:
        """初始化测试替身及其记录状态。"""
        self.response = answer
        self.calls: list[tuple[str, list[RetrievalResult]]] = []

    def answer(self, query: str, results: list[RetrievalResult]) -> str:
        """返回测试预设的回答文本。"""
        self.calls.append((query, results))
        return self.response


def test_rag_maps_only_used_valid_citations_to_source_chunks() -> None:
    """验证 RAG 只将回答实际使用的有效引用映射回来源片段。"""
    results = [_result("chunk-a", score=0.92), _result("chunk-b", score=0.8, page=4)]
    generator = FakeAnswerGenerator("年假为十天。[1] 其他信息见 [2]。")

    response = RAGService(FakeRetriever(results), generator).query("年假有几天？")

    assert response.rejected is False
    assert "[2]" in response.answer
    assert [item.chunk_id for item in response.citations] == ["chunk-a", "chunk-b"]
    assert [item.page_number for item in response.citations] == [3, 4]
    assert [item.chunk.chunk_id for item in response.results] == ["chunk-a", "chunk-b"]
    assert [item.candidate_rank for item in response.results] == [1, 2]


def test_rag_renumbers_sparse_source_ids_to_match_returned_citation_order() -> None:
    """验证稀疏引用编号会按最终顺序重新编号。"""
    results = [_result("first", score=0.95), _result("second", score=0.85, page=4)]
    response = RAGService(
        FakeRetriever(results), FakeAnswerGenerator("只使用第二段证据。[2]")
    ).query("question")

    assert response.answer == "只使用第二段证据。[1]"
    assert [item.chunk_id for item in response.citations] == ["second"]


def test_rag_rejects_low_confidence_without_calling_chat() -> None:
    """验证低置信度检索直接拒答且不调用 Chat。"""
    generator = FakeAnswerGenerator("should not be used [1]")
    retriever = FakeRetriever([_result("low", score=0.24)])

    response = RAGService(retriever, generator, reject_threshold=0.25).query("question")

    assert response.answer == REFUSAL_ANSWER
    assert response.rejected is True
    assert response.citations == ()
    assert generator.calls == []


def test_rag_rejects_empty_retrieval_and_untraceable_answer() -> None:
    """验证空检索和不可追溯回答都会触发拒答。"""
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


def test_rag_rejects_invalid_number_even_when_a_valid_number_is_present() -> None:
    """验证回答含错误数值时不能因另一个正确数值而通过。"""
    results = [_result("chunk-a", score=0.92)]
    response = RAGService(
        FakeRetriever(results),
        FakeAnswerGenerator("有一项可追溯的说法。[1]，另一个来源不存在。[99]"),
    ).query("question")

    assert response.rejected is True
    assert response.answer == REFUSAL_ANSWER
    assert response.citations == ()


def test_rag_refusal_phrase_takes_precedence_over_citation_numbers() -> None:
    """验证明确拒答文本优先于回答中的引用编号。"""
    response = RAGService(
        FakeRetriever([_result("chunk-a", score=0.92)]),
        FakeAnswerGenerator(f"{REFUSAL_ANSWER} [1]"),
    ).query("question")

    assert response.rejected is True
    assert response.answer == REFUSAL_ANSWER
    assert response.citations == ()


def test_rag_rejects_natural_language_refusal_with_a_citation() -> None:
    """验证自然语言明确拒答即使带引用也应拒绝返回。"""
    response = RAGService(
        FakeRetriever([_result("chunk-a", score=0.92)]),
        FakeAnswerGenerator(
            "根据当前知识库，无法确认损坏赔偿金额；资料未列出具体金额。[1]"
        ),
    ).query("损坏赔偿金额是多少？")

    assert response.rejected is True
    assert response.answer == REFUSAL_ANSWER
    assert response.citations == ()


@pytest.mark.parametrize(
    "query",
    [
        "文档有写维修费用吗？",
        "维修费用具体是多少？文档有写吗？",
    ],
)
def test_rag_explicit_refusal_takes_priority_for_presence_and_mixed_questions(
    query: str,
) -> None:
    """验证明确拒答不会被同一答案里的“未列出”事实覆盖。"""

    response = RAGService(
        FakeRetriever([_result("fee", score=0.92)]),
        FakeAnswerGenerator(
            "根据当前知识库，我无法确认这个问题。资料未列出维修费用。[1]"
        ),
    ).query(query)

    assert response.rejected is True
    assert response.answer == REFUSAL_ANSWER
    assert response.citations == ()


def test_rag_keeps_supported_negative_facts_that_are_not_refusals() -> None:
    """验证证据支持的否定事实不会被误判为拒答。"""
    response = RAGService(
        FakeRetriever([_result("chunk-a", score=0.92)]),
        FakeAnswerGenerator("已被其他读者预约的图书不能续借。[1]"),
    ).query("已预约的图书可以续借吗？")

    assert response.rejected is False
    assert response.answer == "已被其他读者预约的图书不能续借。[1]"
    assert [citation.chunk_id for citation in response.citations] == ["chunk-a"]


def test_rag_answers_when_evidence_says_a_detail_is_not_listed() -> None:
    """验证文档明确记载信息未列出时仍可引用回答。"""
    evidence = RetrievalResult(
        Chunk(
            chunk_id="fee-policy",
            document_id="doc-1",
            content="本规定未列出维修费用。",
            file_name="equipment-rules.txt",
            page_number=None,
            chunk_index=0,
        ),
        score=0.92,
    )
    response = RAGService(
        FakeRetriever([evidence]),
        FakeAnswerGenerator("本规定未列出维修费用。[1]"),
    ).query("本规定是否列出维修费用？")

    assert response.rejected is False
    assert response.answer == "本规定未列出维修费用。[1]"
    assert len(response.citations) == 1
    assert response.citations[0].text == evidence.chunk.content


def test_rag_answers_synonymous_document_presence_question() -> None:
    """验证“文档有写吗”这类同义问法仍可引用否定事实。"""

    evidence = RetrievalResult(
        Chunk(
            chunk_id="fee-policy",
            document_id="doc-1",
            content="本规定未列出维修费用。",
            file_name="equipment-rules.txt",
            page_number=None,
            chunk_index=0,
        ),
        score=0.92,
    )
    retriever = FakeRetriever([evidence])
    response = RAGService(
        retriever,
        FakeAnswerGenerator("本规定未列出维修费用。[1]"),
    ).query("文档有写维修费用吗？")

    assert response.rejected is False
    assert response.answer == "本规定未列出维修费用。[1]"
    assert response.citations[0].text == evidence.chunk.content
    assert retriever.calls[0] == ("文档有写维修费用吗？", 10)


def test_rag_still_rejects_a_missing_specific_value() -> None:
    """验证问题只索取文档缺失的具体值时系统正确拒答。"""
    evidence = RetrievalResult(
        Chunk(
            chunk_id="fee-policy",
            document_id="doc-1",
            content="本规定未列出维修费用。",
            file_name="equipment-rules.txt",
            page_number=None,
            chunk_index=0,
        ),
        score=0.92,
    )
    response = RAGService(
        FakeRetriever([evidence]),
        FakeAnswerGenerator("资料未列出维修费用的具体数额。[1]"),
    ).query("维修费用具体是多少？")

    assert response.rejected is True
    assert response.answer == REFUSAL_ANSWER
    assert response.citations == ()


def test_rag_answers_supported_absence_part_of_a_mixed_question() -> None:
    """验证混合询问可回答已证实的信息状态且不编造缺失数值。"""

    evidence = RetrievalResult(
        Chunk(
            chunk_id="fee-policy",
            document_id="doc-1",
            content="本规定未列出维修费用。",
            file_name="equipment-rules.txt",
            page_number=None,
            chunk_index=0,
        ),
        score=0.92,
    )
    response = RAGService(
        FakeRetriever([evidence]),
        FakeAnswerGenerator("文档未列出维修费用的具体数额，因此没有具体金额可提供。[1]"),
    ).query("维修费用具体是多少？文档有写吗？")

    assert response.rejected is False
    assert len(response.citations) == 1
    assert response.citations[0].text == evidence.chunk.content


@pytest.mark.parametrize("top_k", [1, 5, 20])
@pytest.mark.parametrize("presence_question", [False, True])
def test_rag_caps_model_context_and_returned_retrievals_to_top_k(
    top_k: int, presence_question: bool
) -> None:
    """验证普通与是否记载问题的模型上下文和 API 结果都受 Top-K 限制。"""

    results = []
    promoted_rank = top_k + 5
    for index in range(25):
        is_promoted_evidence = presence_question and index + 1 == promoted_rank
        content = (
            "本规定未列出维修费用。"
            if is_promoted_evidence
            else f"无关片段 {index + 1}。"
        )
        chunk = Chunk(
            chunk_id=f"chunk-{index + 1}",
            document_id="doc-1",
            content=content,
            file_name="policy.txt",
            page_number=None,
            chunk_index=index,
        )
        results.append(RetrievalResult(chunk=chunk, score=0.99 - index / 1000))

    query = "文档有写维修费用吗？" if presence_question else "年假政策是什么？"
    generator = FakeAnswerGenerator(f"答案。[{top_k}]")
    retriever = FakeRetriever(results)
    response = RAGService(retriever, generator).query(query, top_k=top_k)

    expected_candidate_count = top_k + 5 if presence_question else top_k
    assert retriever.calls == [(query, expected_candidate_count)]
    assert len(generator.calls[0][1]) == top_k
    assert len(response.results) == top_k
    if presence_question:
        assert response.results[-1].chunk.content == "本规定未列出维修费用。"
        assert response.results[-1].candidate_rank == promoted_rank
        assert response.citations[-1].text == "本规定未列出维修费用。"


def test_rag_validates_query_top_k_and_threshold() -> None:
    """验证 RAG 拒绝无效问题、Top-K 和拒答阈值。"""
    with pytest.raises(ValueError, match="between -1 and 1"):
        RAGService(FakeRetriever([]), FakeAnswerGenerator(""), reject_threshold=1.1)
    service = RAGService(FakeRetriever([]), FakeAnswerGenerator(""))
    with pytest.raises(ValueError, match="Query cannot be empty"):
        service.query("  ")
    with pytest.raises(ValueError, match="top_k"):
        service.query("question", top_k=0)


def test_chat_client_sends_ordered_source_metadata_and_grounding_instructions() -> None:
    """验证 Chat 请求按检索顺序发送来源元数据和证据约束。"""
    class Endpoint:
        request: dict[str, object] | None = None

        def create(self, **kwargs: object) -> SimpleNamespace:
            """根据测试输入生成预设的模型响应。"""
            self.request = kwargs
            message = SimpleNamespace(content=json.dumps({
                "refused": False,
                "passage_ids": ["1:1"],
            }))
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
    assert answer == "Evidence text for first[1]。"
    assert request is not None
    assert request["model"] == "test-chat"
    assert request["temperature"] == 0
    messages = request["messages"]
    assert "untrusted quoted data" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["question"] == "年假几天？"
    assert [source["chunk_id"] for source in payload["evidence"]] == ["first", "second"]
    assert [source["id"] for source in payload["evidence"]] == [1, 2]
    assert "Answer concisely" in messages[0]["content"]
    assert SYSTEM_PROMPT == messages[0]["content"]


def test_chat_client_requires_credentials_and_nonempty_output() -> None:
    """验证 Chat 客户端检查凭据并拒绝空模型回答。"""
    with pytest.raises(ValueError, match="TRACERAG_API_KEY"):
        ChatClient(api_key=None, base_url=None, model="chat")

    class EmptyEndpoint:
        def create(self, **kwargs: object) -> SimpleNamespace:
            """根据测试输入生成预设的模型响应。"""
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
