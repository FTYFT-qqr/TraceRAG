"""验证检索验收脚本会把作答、预期事实和引用质量纳入通过条件。"""

from types import SimpleNamespace

from app.models import Chunk, Citation, RetrievalResult
from scripts.evaluate_retrieval import _acceptance_passes, _assess_case


def _case() -> dict[str, object]:
    """构造回答质量评估用的固定样例。"""
    return {
        "id": "LAB-05",
        "answerable": True,
        "expected_source": "lab_equipment.txt",
        "expected_evidence": "本规定未列出维修费用",
        "expected_answer_facts": [["未列出维修费用", "没有列出维修费用"]],
    }


def _response(
    *,
    answer: str,
    rejected: bool = False,
    citation_file_name: str = "lab_equipment.txt",
    chunk_text: str = "本规定未列出维修费用。",
    citation_text: str | None = None,
) -> SimpleNamespace:
    """构造可控制引用和拒答状态的模拟响应。"""
    chunk = Chunk(
        chunk_id="fee-chunk",
        document_id="lab-document",
        content=chunk_text,
        file_name="lab_equipment.txt",
        page_number=None,
        chunk_index=0,
    )
    citations = (
        Citation(
            chunk_id=chunk.chunk_id,
            file_name=citation_file_name,
            page_number=None,
            text=citation_text if citation_text is not None else chunk.content,
        ),
    ) if not rejected else ()
    return SimpleNamespace(
        answer=answer,
        rejected=rejected,
        citations=citations,
        results=(RetrievalResult(chunk=chunk, score=0.91),),
    )


def test_answerable_case_passes_only_with_correct_grounded_answer() -> None:
    """验证有答案的问题必须满足事实、引用和证据约束才能通过。"""
    case = _case()
    outcome = _assess_case(
        case,
        _response(answer="本规定未列出维修费用。[1]"),
    )

    assert outcome["expected_evidence_hit"] is True
    assert outcome["normal_answer"] is True
    assert outcome["answer_correct"] is True
    assert outcome["citations_traceable"] is True
    assert outcome["citation_supports_expected_evidence"] is True
    assert outcome["quality_pass"] is True


def test_cited_source_attribution_does_not_require_manual_review() -> None:
    """验证引用来源的归因短语不会被误判为额外事实。"""

    case = _case()
    outcome = _assess_case(
        case,
        _response(
            answer=(
                "星屿实验室设备借用与归还规定明确说明，"
                "本规定未列出维修费用。[1]"
            )
        ),
    )

    assert outcome["manual_review_status"] == "自动核验通过"
    assert outcome["unverified_claims"] == []
    assert outcome["quality_pass"] is True


def test_evidence_ranked_after_five_is_not_reported_as_a_top_five_hit() -> None:
    """验证候选集第六名之后的证据不计入 Top-5 命中。"""

    case = _case()
    response = _response(answer="本规定未列出维修费用。[1]")
    fillers = [
        RetrievalResult(
            chunk=Chunk(
                chunk_id=f"filler-{index}",
                document_id="other-document",
                content=f"不相关片段 {index}",
                file_name="other.txt",
                page_number=None,
                chunk_index=index,
            ),
            score=0.99 - index / 1000,
        )
        for index in range(6)
    ]
    response.results = tuple([*fillers, *response.results])

    outcome = _assess_case(case, response)

    assert outcome["expected_evidence_hit"] is False
    assert outcome["expected_evidence_rank"] == 7
    assert outcome["quality_pass"] is True


def test_evidence_hit_but_rejected_answer_fails_acceptance() -> None:
    """验证检索命中但系统拒答时整体验收仍失败。"""
    case = _case()
    outcome = _assess_case(
        case,
        _response(answer="根据当前知识库，我无法确认这个问题。", rejected=True),
    )

    assert outcome["expected_evidence_hit"] is True
    assert outcome["quality_pass"] is False
    assert _acceptance_passes([case], [outcome]) is False


def test_wrong_answer_or_mismatched_citation_fails_acceptance() -> None:
    """验证错误答案或引用来源不匹配都会导致验收失败。"""
    case = _case()
    wrong_answer = _assess_case(
        case,
        _response(answer="维修费用为500元。[1]"),
    )
    wrong_citation = _assess_case(
        case,
        _response(
            answer="本规定未列出维修费用。[1]",
            citation_file_name="other.txt",
        ),
    )

    assert wrong_answer["expected_evidence_hit"] is True
    assert wrong_answer["answer_correct"] is False
    assert wrong_answer["quality_pass"] is False
    assert wrong_citation["answer_correct"] is True
    assert wrong_citation["citations_traceable"] is False
    assert wrong_citation["quality_pass"] is False
    assert _acceptance_passes([case], [wrong_answer]) is False
    assert _acceptance_passes([case], [wrong_citation]) is False


def test_numeric_fact_uses_full_value_and_rejects_fourteen_for_four() -> None:
    """验证数值匹配遵守完整边界，不能把十四小时算作四小时。"""

    case = {
        **_case(),
        "expected_evidence": "蓝鲸X7每次借用最长4小时",
        "expected_answer_facts": [["4小时", "四小时"]],
    }
    outcome = _assess_case(
        case,
        _response(
            answer="蓝鲸X7每次借用最长14小时。[1]",
            chunk_text="蓝鲸X7每次借用最长4小时。",
        ),
    )

    assert outcome["expected_fact_phrases_match"] is False
    assert outcome["unsupported_numeric_claims"] == ["14小时"]
    assert outcome["answer_correct"] is False
    assert outcome["quality_pass"] is False
    assert _acceptance_passes([case], [outcome]) is False


def test_opposite_polarity_does_not_pass_by_containing_expected_phrase() -> None:
    """验证否定反转不会因答案含预期短语而通过。"""

    case = _case()
    outcome = _assess_case(
        case,
        _response(answer="并非未列出维修费用，维修费用为500元。[1]"),
    )

    assert outcome["expected_fact_phrases_match"] is True
    assert outcome["contradictory_fact"] is True
    assert outcome["answer_correct"] is False
    assert outcome["quality_pass"] is False
    assert _acceptance_passes([case], [outcome]) is False


def test_unverified_extra_claim_in_same_conjoined_sentence_requires_review() -> None:
    """验证与正确事实用“且”连接的额外断言也须人工复核。"""

    case = _case()
    outcome = _assess_case(
        case,
        _response(answer="本规定未列出维修费用且所有维修均禁止开展。[1]"),
    )

    assert outcome["expected_fact_phrases_match"] is True
    assert outcome["manual_review_status"] == "待人工复核"
    assert outcome["unverified_claims"] == ["所有维修均禁止开展"]
    assert outcome["answer_correct"] is None
    assert outcome["quality_pass"] is False
    assert _acceptance_passes([case], [outcome]) is False


def test_chinese_fourteen_hours_cannot_match_expected_four_hours() -> None:
    """验证中文数字按完整数值核对，不能把十四小时认作四小时。"""

    case = {
        **_case(),
        "expected_evidence": "蓝鲸X7每次借用最长4小时",
        "expected_answer_facts": [["4小时", "四小时"]],
    }
    outcome = _assess_case(
        case,
        _response(
            answer="蓝鲸X7每次借用最长十四小时。[1]",
            chunk_text="蓝鲸X7每次借用最长4小时。",
        ),
    )

    assert outcome["expected_fact_phrases_match"] is False
    assert outcome["unsupported_numeric_claims"] == ["十四小时"]
    assert outcome["answer_correct"] is False
    assert outcome["quality_pass"] is False


def test_chinese_and_arabic_equivalent_numbers_match_with_units() -> None:
    """验证中文与阿拉伯数字的等值表达可以匹配并由引用支持。"""

    case = {
        **_case(),
        "expected_evidence": "蓝鲸X7每次借用最长4小时",
        "expected_answer_facts": [["4小时", "四小时"]],
    }
    outcome = _assess_case(
        case,
        _response(
            answer="蓝鲸X7每次借用最长四小时。[1]",
            chunk_text="蓝鲸X7每次借用最长4小时。",
        ),
    )

    assert outcome["expected_fact_phrases_match"] is True
    assert outcome["unsupported_numeric_claims"] == []
    assert outcome["quality_pass"] is True


def test_unanswerable_quality_requires_refusal_text_and_consistent_status() -> None:
    """验证无答案验收同时核对拒答标记、拒答文字和引用状态。"""

    case = {**_case(), "answerable": False}
    correct = SimpleNamespace(
        answer="根据当前知识库，我无法确认这个问题。",
        rejected=True,
        citations=(),
        results=(),
    )
    inconsistent_responses = [
        SimpleNamespace(answer="维修费用未列出。", rejected=True, citations=(), results=()),
        SimpleNamespace(
            answer="根据当前知识库，我无法确认这个问题。",
            rejected=False,
            citations=(),
            results=(),
        ),
        SimpleNamespace(
            answer="根据当前知识库，我无法确认这个问题。[1]",
            rejected=True,
            citations=(),
            results=(),
        ),
    ]

    passed = _assess_case(case, correct)
    failed = [_assess_case(case, response) for response in inconsistent_responses]

    assert passed["refusal_text_present"] is True
    assert passed["quality_pass"] is True
    assert all(result["quality_pass"] is False for result in failed)
    assert all(result["manual_review_status"] == "自动核验失败" for result in failed)


def test_single_question_execution_error_fails_acceptance() -> None:
    """确认单题执行错误不能计入整体验收通过。"""

    case = _case()
    errored = _assess_case(case, None, error="TimeoutError")

    assert errored["error"] == "TimeoutError"
    assert errored["quality_pass"] is False
    assert _acceptance_passes([case], [errored]) is False


def test_multiple_expected_evidence_spans_require_all_citations() -> None:
    """确认跨 Chunk 题必须召回并引用每条独立预期证据。"""

    case = {
        "id": "LAB-V2-11",
        "answerable": True,
        "expected_source": "lab_equipment.txt",
        "expected_evidence": ["蓝鲸X7每次借用最长4小时", "校准期间设备标记为不可借用"],
        "expected_answer_facts": [["最长4小时"], ["不可借用"]],
    }
    first = Chunk("duration", "lab", "蓝鲸X7每次借用最长4小时。", "lab_equipment.txt", None, 0)
    second = Chunk("calibration", "lab", "季度校准期间设备标记为不可借用。", "lab_equipment.txt", None, 1)
    response = SimpleNamespace(
        answer="蓝鲸X7最长4小时。[1]季度校准期间不可借用。[2]",
        rejected=False,
        citations=(
            Citation("duration", "lab_equipment.txt", None, first.content),
            Citation("calibration", "lab_equipment.txt", None, second.content),
        ),
        results=(
            RetrievalResult(first, 0.9, candidate_rank=1),
            RetrievalResult(second, 0.8, candidate_rank=2),
        ),
    )

    complete = _assess_case(case, response)
    response.citations = response.citations[:1]
    partial = _assess_case(case, response)

    assert complete["expected_evidence_rank"] == 2
    assert complete["expected_evidence_hit"] is True
    assert complete["citation_supports_expected_evidence"] is True
    assert complete["quality_pass"] is True
    assert partial["citation_supports_expected_evidence"] is False
    assert partial["quality_pass"] is False


def test_shared_negation_covers_coordinated_items_but_not_a_contrast() -> None:
    """验证并列对象可共享否定词，转折后的肯定说法不能误算否定。"""

    case = {
        **_case(),
        "expected_evidence": "不提供图书配送",
        "expected_answer_facts": [["不提供图书配送"], ["不提供读者寄存柜"]],
    }
    supported = _assess_case(
        case,
        _response(
            answer="不提供图书配送或读者寄存柜。[1]",
            chunk_text="馆内不提供图书配送或读者寄存柜服务。",
        ),
    )
    contrasted = _assess_case(
        case,
        _response(
            answer="不提供图书配送或读者寄存柜。[1]",
            chunk_text="馆内不提供图书配送，但提供读者寄存柜。",
        ),
    )

    assert supported["quality_pass"] is True
    assert contrasted["expected_facts_supported_by_citations"] is False
    assert contrasted["quality_pass"] is False


def test_short_confirmation_must_match_supported_polarity() -> None:
    """验证简短确认语只有与问题和引用中的否定事实一致时才通过。"""

    case = {
        **_case(),
        "query": "已预约的图书还可以续借吗？",
        "expected_evidence": "已预约的图书不能续借",
        "expected_answer_facts": [["不能续借"]],
    }
    grounded = _response(
        answer="不可以。已预约的图书不能续借。[1]",
        chunk_text="已预约的图书不能续借。",
    )
    contradictory = _response(
        answer="可以。已预约的图书不能续借。[1]",
        chunk_text="已预约的图书不能续借。",
    )

    assert _assess_case(case, grounded)["quality_pass"] is True
    assert _assess_case(case, contradictory)["quality_pass"] is False


def test_unit_correction_requires_cited_hour_limit() -> None:
    """验证“几天”可按文档纠正为小时，但仍须给出可核对的时长。"""

    case = {
        **_case(),
        "query": "设备最多能借几天？",
        "expected_evidence": "设备每次借用最长4小时",
        "expected_answer_facts": [["最长4小时"]],
    }
    grounded = _assess_case(
        case,
        _response(
            answer="每次借用最长4小时，不是按天计算。[1]",
            chunk_text="设备每次借用最长4小时。",
        ),
    )
    wrong_number = _assess_case(
        case,
        _response(
            answer="每次借用最长14小时，不是按天计算。[1]",
            chunk_text="设备每次借用最长4小时。",
        ),
    )

    assert grounded["quality_pass"] is True
    assert wrong_number["quality_pass"] is False


def test_extra_claim_joined_with_and_remains_unverified() -> None:
    """验证正确事实后用“并”追加的无依据说法仍须人工复核。"""

    outcome = _assess_case(
        _case(),
        _response(answer="本规定未列出维修费用并禁止所有维修。[1]"),
    )

    assert outcome["quality_pass"] is False
    assert outcome["unverified_claims"] == ["禁止所有维修"]
