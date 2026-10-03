"""覆盖整改记录 02 的数值、完整断言及逐句引用误通过边界。"""

from types import SimpleNamespace

import pytest

from app.evaluation import assess_case
from app.models import Chunk, Citation, RetrievalResult
from app.rag import RAGService


def _assessed(source: str, fact: str, answer: str) -> dict:
    """用来源有效的固定响应检验评分，不访问模型或本地知识库。"""

    chunk = Chunk("fixed", "doc", source, "audit.txt", None, 0)
    case = {
        "id": "fixed", "query": "请根据文档回答。", "answerable": True,
        "expected_source": chunk.file_name, "expected_evidence": source,
        "expected_answer_facts": [[fact]],
    }
    response = SimpleNamespace(
        answer=answer, rejected=False,
        citations=(Citation(chunk.chunk_id, chunk.file_name, None, source),),
        results=(RetrievalResult(chunk, 0.9, 1),),
    )
    return assess_case(case, response)


@pytest.mark.parametrize("source_value,answer_value,equivalent", [
    ("40小时", "4.0小时", False),
    ("四十小时", "四点零小时", False),
    ("4小时", "4.000小时", True),
    ("四小时", "四点零小时", True),
    ("4.5小时", "四点五小时", True),
    ("0.04小时", "0.4小时", False),
    ("0.4小时", "4小时", False),
    ("4小时", "14.0小时", False),
    ("4小时", "-4小时", False),
    ("-4小时", "负四点零小时", True),
    ("-4小时", "+4小时", False),
    ("4小时", "−4.0小时", False),
    ("+4小时", "4.0小时", True),
    ("40小时", "4，0小时", False),
    ("40小时", "4 0小时", False),
    ("40", "4.0", False),
    ("40", "四点零", False),
    ("4", "十四", False),
    ("4", "负四", False),
    ("-4", "负四点零", True),
    ("4小时", "- 4小时", False),
    ("-4小时", "负 四点零小时", True),
    ("40小时", "4， 0小时", False),
    ("0.5小时", ".5小时", True),
    ("0.5小时", "-.5小时", False),
    ("-0.5小时", "-.5小时", True),
])
def test_complete_numeric_values_decide_quality(
    source_value: str, answer_value: str, equivalent: bool
) -> None:
    """确认等值整数小数可通过，错误量级、数字合并或负号直接失败。"""

    outcome = _assessed(
        f"设备每次借用最长{source_value}。", source_value,
        f"设备每次借用最长{answer_value}。[1]",
    )
    assert outcome["quality_pass"] is equivalent
    assert outcome["answer_correct"] is equivalent
    if not equivalent:
        assert outcome["manual_review_status"] == "自动核验失败"


@pytest.mark.parametrize("connector", ["并", "并且", "且", "以及", "而且", "同时", "和"])
def test_extra_assertion_after_matching_fact_requires_review(connector: str) -> None:
    """正确短语后追加的无来源断言必须复核，不能因部分命中自动通过。"""

    outcome = _assessed(
        "本规定未列出维修费用。", "未列出维修费用",
        f"本规定未列出维修费用{connector}支持终身免费维修。[1]",
    )
    assert outcome["quality_pass"] is False
    assert outcome["answer_correct"] is None
    assert outcome["manual_review_status"] == "待人工复核"
    assert outcome["unverified_claims"]


@pytest.mark.parametrize("answer", [
    "校准期间设备不可借用是不正确的。[1]",
    "校准期间设备不可借用是错误的。[1]",
    "校准期间设备不可借用不属实。[1]",
    "校准期间设备不可借用，但校准期间设备可以借用。[1]",
    "校准期间设备可借用。[1]",
])
def test_explicit_polarity_reversal_fails_automatically(answer: str) -> None:
    """确认后置否定及转折后的相反事实直接失败。"""

    outcome = _assessed("校准期间设备不可借用。", "不可借用", answer)
    assert outcome["contradictory_fact"] is True
    assert outcome["answer_correct"] is False
    assert outcome["manual_review_status"] == "自动核验失败"


def test_supported_additional_assertion_can_pass() -> None:
    """确认引用明确支持的并列断言不会因新规则而被误拒绝。"""

    outcome = _assessed(
        "本规定未列出维修费用。校准期间设备不可借用。", "未列出维修费用",
        "本规定未列出维修费用且校准期间设备不可借用。[1]",
    )
    assert outcome["quality_pass"] is True


def _citation_case(answer: str) -> tuple[dict, SimpleNamespace]:
    """构造两段各支持一项事实的响应，以检验实际引用编号。"""

    chunks = [
        Chunk("duration", "doc", "蓝鲸X7每次借用最长4小时。", "audit.txt", None, 0),
        Chunk("calibration", "doc", "校准期间设备不可借用。", "audit.txt", None, 1),
    ]
    case = {
        "id": "citations", "query": "设备借用上限和校准期间的借用规则？",
        "answerable": True, "expected_source": "audit.txt",
        "expected_evidence": [chunk.content for chunk in chunks],
        "expected_answer_facts": [["最长4小时"], ["不可借用"]],
    }
    response = SimpleNamespace(
        answer=answer, rejected=False,
        citations=tuple(Citation(chunk.chunk_id, chunk.file_name, None, chunk.content) for chunk in chunks),
        results=tuple(RetrievalResult(chunk, 0.9, rank) for rank, chunk in enumerate(chunks, 1)),
    )
    return case, response


@pytest.mark.parametrize("answer", [
    "蓝鲸X7每次借用最长4小时。[1]校准期间设备不可借用。[2]",
    "蓝鲸X7每次借用最长4小时[1]。校准期间设备不可借用[2]。",
    "蓝鲸X7每次借用最长4.0小时，校准期间设备不可借用。[1][2]",
])
def test_correct_citation_association_and_multi_source_statement_pass(answer: str) -> None:
    """正确分句引用和同一断言关联多条证据均应通过。"""

    case, response = _citation_case(answer)
    outcome = assess_case(case, response)
    assert outcome["citation_claims_supported"] is True
    assert outcome["quality_pass"] is True


@pytest.mark.parametrize("answer", [
    "蓝鲸X7每次借用最长4小时。[2]校准期间设备不可借用。[1]",
    "蓝鲸X7每次借用最长4小时。[1]校准期间设备不可借用。[1]",
    "蓝鲸X7每次借用最长4小时。[3]校准期间设备不可借用。[2]",
])
def test_incorrect_or_nonexistent_citation_association_fails(answer: str) -> None:
    """证据集合完整也不能掩盖单句引用错配或无效编号。"""

    case, response = _citation_case(answer)
    outcome = assess_case(case, response)
    assert outcome["citation_claims_supported"] is False
    assert outcome["quality_pass"] is False
    assert outcome["manual_review_status"] == "自动核验失败"


@pytest.mark.parametrize("answer", [
    "蓝鲸X7每次借用最长4小时。校准期间设备不可借用。[1][2]",
    "蓝鲸X7每次借用最长4小时。[1]校准期间设备不可借用。",
])
def test_missing_or_ambiguous_sentence_citation_requires_review(answer: str) -> None:
    """漏标引用或跨句尾部统一引用缺乏明确范围时不能自动通过。"""

    case, response = _citation_case(answer)
    outcome = assess_case(case, response)
    assert outcome["citation_claims_supported"] is None
    assert outcome["quality_pass"] is False
    assert outcome["manual_review_status"] == "待人工复核"


def test_citation_swap_is_detected_after_real_rag_renumbering() -> None:
    """通过真实引用重编号链路后，仍能阻止文档中的颠倒引用响应通过。"""

    case, response = _citation_case("")
    retriever = SimpleNamespace(retrieve=lambda query, top_k: list(response.results)[:top_k])
    generator = SimpleNamespace(answer=lambda query, results: "蓝鲸X7每次借用最长4小时。[2]校准期间设备不可借用。[1]")
    actual = RAGService(retriever, generator).query(case["query"])
    assert actual.citations[0].chunk_id == "calibration"
    assert assess_case(case, actual)["quality_pass"] is False


@pytest.mark.parametrize("source,answer", [
    ("下午1点30分至5点。", "下午为1点30分至5点[1]。"),
    ("下午为1点30分至5点。", "下午1点30分至5点[1]。"),
])
def test_declared_fact_alternatives_share_citation_support(source: str, answer: str) -> None:
    """同组已声明等义表达可由原文另一表达支持，完整附加断言仍需核验。"""

    case, response = _citation_case(answer)
    chunk = Chunk("time", "doc", source, "audit.txt", None, 0)
    case["expected_evidence"] = source
    case["expected_answer_facts"] = [["下午1点30分至5点", "下午为1点30分至5点"]]
    response.citations = (Citation(chunk.chunk_id, chunk.file_name, None, source),)
    response.results = (RetrievalResult(chunk, 0.9, 1),)
    outcome = assess_case(case, response)
    assert outcome["citation_claims_supported"] is True
    assert outcome["quality_pass"] is True
    response.answer = answer.replace("[1]", "并提供终身免费服务[1]")
    assert assess_case(case, response)["quality_pass"] is False


def test_equivalent_fact_cannot_borrow_another_statements_source() -> None:
    """等义支持仍限定当前编号，不能借其他句引用中的正确事实。"""

    case, response = _citation_case("下午为1点30分至5点[1]。校准期间设备不可借用[2]。")
    time = Chunk("time", "doc", "下午1点30分至5点。", "audit.txt", None, 2)
    response.citations += (Citation(time.chunk_id, time.file_name, None, time.content),)
    response.results += (RetrievalResult(time, 0.8, 3),)
    case["expected_answer_facts"] = [["下午1点30分至5点", "下午为1点30分至5点"], ["不可借用"]]
    case["expected_evidence"] = [time.content, "校准期间设备不可借用。"]
    outcome = assess_case(case, response)
    assert outcome["citation_claims_supported"] is False
    assert outcome["quality_pass"] is False


def test_unrelated_document_rule_cannot_pass_even_when_every_quote_is_supported() -> None:
    """本题证据完整也不能掩盖引用另一文档的相似条款。"""

    case, response = _citation_case("蓝鲸X7每次借用最长4小时[1]。校准期间设备不可借用[2]。图书馆不提供配送[3]。")
    unrelated = Chunk("library", "library", "图书馆不提供配送。", "library.txt", None, 0)
    response.citations += (Citation(unrelated.chunk_id, unrelated.file_name, None, unrelated.content),)
    response.results += (RetrievalResult(unrelated, 0.5, 3),)
    outcome = assess_case(case, response)
    assert outcome["citations_traceable"] is True
    assert outcome["citation_claims_supported"] is True
    assert outcome["unexpected_citation_sources"] == ["library.txt"]
    assert outcome["citation_source_scope_valid"] is False
    assert outcome["quality_pass"] is False
    assert outcome["manual_review_status"] == "自动核验失败"
