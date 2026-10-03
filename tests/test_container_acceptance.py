"""防止容器工程验收把缺少事实、错误来源或有引用拒答判为通过。"""

from copy import deepcopy

from scripts.container_acceptance import answer_passes


def supported_case() -> tuple[dict, dict]:
    """构造有证据的否定事实，模拟正式接口返回结构。"""
    question = {
        "id": "NEGATIVE", "query": "图书馆提供配送服务吗？",
        "answerable": True,
        "expected_answer_facts": [["不提供配送", "没有配送服务"]],
        "expected_source": "library_service.txt",
        "expected_evidence": "图书馆没有配送服务",
    }
    response = {
        "answer": "图书馆没有配送服务[1]。", "rejected": False,
        "citations": [{"file_name": "library_service.txt", "chunk_id": "chunk-1", "page_number": None, "text": "图书馆没有配送服务。"}],
        "retrievals": [{"chunk_id": "chunk-1", "file_name": "library_service.txt", "page_number": None, "text": "图书馆没有配送服务。", "score": 1.0, "candidate_rank": 1}],
    }
    return question, response


def test_negative_fact_is_an_answer() -> None:
    """知识库明确记载的否定事实必须被验收为有证据回答。"""
    question, response = supported_case()
    assert answer_passes(question, response)


def test_missing_answer_fact_fails() -> None:
    """返回成功且带引用，也不能忽略题目要求的事实。"""
    question, response = supported_case()
    response["answer"] = "图书馆有服务台[1]。"
    assert not answer_passes(question, response)


def test_wrong_source_or_unknown_chunk_fails() -> None:
    """引用必须来自预期文件，并确实存在于本次检索候选。"""
    question, response = supported_case()
    wrong_source = deepcopy(response)
    wrong_source["citations"][0]["file_name"] = "other.txt"
    assert not answer_passes(question, wrong_source)
    response["citations"][0]["chunk_id"] = "unknown"
    assert not answer_passes(question, response)


def test_refusing_supported_fact_fails() -> None:
    """有证据问题即使回答含关键词，拒答标记也不能被误判为通过。"""
    question, response = supported_case()
    response["rejected"] = True
    assert not answer_passes(question, response)


def test_citation_without_answer_evidence_fails() -> None:
    """文件名和片段编号正确时，引用正文仍必须支持要求的事实。"""
    question, response = supported_case()
    response["citations"][0]["text"] = "图书馆设有服务台。"
    assert not answer_passes(question, response)


def test_missing_evidence_requires_empty_citations() -> None:
    """无证据问题必须明确拒答，而且不能保留看似支持答案的引用。"""
    question = {"id": "NONE", "answerable": False}
    response = {"answer": "根据当前知识库，我无法确认这个问题。", "rejected": True, "citations": [], "retrievals": []}
    assert answer_passes(question, response)
    assert not answer_passes(question, dict(response, rejected=False))
    assert not answer_passes(question, dict(response, answer="这是肯定的。"))
    citation = {"chunk_id": "chunk-1", "file_name": "other.txt", "page_number": None, "text": "无关资料"}
    assert not answer_passes(question, dict(response, citations=[citation]))


def test_coordinated_negative_uses_shared_quality_rules() -> None:
    """“不提供A或B”应覆盖两个否定事实，避免验收自身出现误拒答判断。"""
    question, response = supported_case()
    text = "馆内不提供图书配送或读者寄存柜服务"
    question.update(
        query="图书馆提供图书配送或读者寄存柜吗？", expected_evidence=text,
        expected_answer_facts=[["不提供图书配送"], ["不提供读者寄存柜"]],
    )
    response["answer"] = text + "[1]。"
    response["citations"][0]["text"] = text + "。"
    response["retrievals"][0]["text"] = text + "。"
    assert answer_passes(question, response)
