"""提供 RAG 评测脚本共用的回答、引用和拒答质量判定。"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from app.citations import cited_statements
from app.rag import has_explicit_refusal


_ANSWER_PUNCTUATION = re.compile(r"[\s，。！？、；：,!?;:（）()\[\]{}“”\"'‘’]+")
_ARABIC_NUMBER = re.compile(r"(?<![\d.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![\d.])")
_CHINESE_NUMBER_CHARS = "零〇一二两三四五六七八九十百千万亿"
_CHINESE_NUMBER_PATTERN = re.compile(
    rf"[负正]?[{_CHINESE_NUMBER_CHARS}]+(?:点[零〇一二两三四五六七八九]+)?"
)
_CLAIM_SEPARATOR = re.compile(
    r"[，,；;！？!?]|\.(?!\d)|。|"
    r"(?:但是|不过|而且|并且|此外|另外|因此|所以|然而|可是|以及|同时|但|却|且|并(?!非)|"
    r"(?:和|及|还)(?=支持|提供|允许|禁止|保证|可|能|会|免费))"
)
_COORDINATED_NEGATIONS = ("不会要求", "未列出", "不提供", "不得")
_NUMERIC_VALUE = (
    rf"(?:[+\-−－负正]?(?:\d+(?:\.\d+)?|\.\d+|[{_CHINESE_NUMBER_CHARS}]+(?:点[零〇一二两三四五六七八九]+)?))"
)
_NUMERIC_UNIT_CLAIM = re.compile(
    rf"(?<![\d.\-+负正{_CHINESE_NUMBER_CHARS}]){_NUMERIC_VALUE}\s*"
    r"(?:个工作日|工作日|小时|分钟|秒钟?|天|年|个月|月|元|块钱|块|%|公斤|千克|千瓦|摄氏度|厘米|毫米|米|次|本|岁|点|号|人|倍)"
    rf"(?![\d{_CHINESE_NUMBER_CHARS}])"
)
_CONTRADICTORY_PREFIX = re.compile(r"(?:并非|并不是|并没有|并未|并不|不是|不等于)")
_CONTRADICTORY_SUFFIX = re.compile(
    r".{0,4}(?:不正确|不对|错误的?|不成立|不属实|不是事实|不是真的|假的)"
)
_RESIDUAL_ASSERTION = re.compile(r"支持|提供|允许|禁止|保证|终身|免费|正确|错误|属实|可以|能够")
_NEGATIVE_TO_POSITIVE = {
    "未列出": ("列出",), "没有列出": ("列出",), "不提供": ("提供",),
    "未提供": ("提供",), "不会": ("会",), "不能": ("能", "可以"),
    "不可": ("可", "可以"), "不得": ("可以",),
}
_NON_FACTUAL_LEADS = (
    "根据证据",
    "根据资料",
    "根据文档",
    "根据手册",
    "根据规定",
    "证据明确指出",
    "文档明确写明",
    "星屿实验室设备借用与归还规定明确说明",
    "资料显示",
    "手册显示",
)
_INCOMPLETE_CLAIM = re.compile(
    r"^(?:没有|未|不|不能|无法)(?:写|列出|提供|说明|提及|记载|确认|回答)?$"
)


def _normalize_answer(text: str) -> str:
    """统一中文数字并移除常见标点，便于按完整数值核对事实。"""

    normalized = _CITATION_MARKER.sub("", text).replace("−", "-").replace("－", "-")
    normalized = re.sub(r"([+\-负正])\s+(?=[\d零〇一二两三四五六七八九十百千万亿])", r"\1", normalized)
    normalized = _CHINESE_NUMBER_PATTERN.sub(_canonical_chinese_number, normalized)
    normalized = re.sub(r"负(?=\d)", "-", normalized)
    normalized = re.sub(r"正(?=\d)", "+", normalized)
    normalized = _ARABIC_NUMBER.sub(_canonical_arabic_number, normalized)
    # 数字之间的逗号、空格等保留边界，不能把“4，0”清理成“40”。
    normalized = _ANSWER_PUNCTUATION.sub(
        lambda match: "|"
        if match.start() > 0 and match.end() < len(normalized)
        and normalized[match.start() - 1].isdigit() and normalized[match.end()].isdigit()
        else "",
        normalized,
    )
    return re.sub(r"(?<!\d)\.|\.(?!\d)", "", normalized).casefold()


def _canonical_arabic_number(match: re.Match[str]) -> str:
    """用十进制完整值统一整数、小数和正负号，避免浮点舍入与量级混淆。"""

    value = Decimal(match.group())
    if value == 0:
        return "0"
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _canonical_chinese_number(match: re.Match[str]) -> str:
    """将常用中文数词转为阿拉伯数字，保留点数及逐位年份写法。"""

    token = match.group()
    sign = "-" if token.startswith("负") else ""
    token = token.removeprefix("负").removeprefix("正")
    digit_values = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if "点" in token:
        integer, decimal = token.split("点", maxsplit=1)
        whole = _parse_chinese_integer(integer)
        fraction = "".join(str(digit_values[character]) for character in decimal)
        return f"{sign}{whole}.{fraction}"
    if not any(unit in token for unit in "十百千万亿"):
        return sign + "".join(str(digit_values[character]) for character in token)
    return sign + str(_parse_chinese_integer(token))


def _parse_chinese_integer(token: str) -> int:
    """解析含十、百、千、万、亿单位的常用中文整数。"""

    digit_values = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    # 先按大单位递归拆分，兼容“一万亿”这类连续量级写法。
    if "亿" in token:
        left, right = token.split("亿", maxsplit=1)
        return (_parse_chinese_integer(left) or 1) * 100_000_000 + (
            _parse_chinese_integer(right) if right else 0
        )
    if "万" in token:
        left, right = token.split("万", maxsplit=1)
        return _parse_chinese_integer(left) * 10_000 + (
            _parse_chinese_integer(right) if right else 0
        )

    small_units = {"十": 10, "百": 100, "千": 1000}
    total = section = number = 0
    for character in token:
        if character in digit_values:
            number = digit_values[character]
        elif character in small_units:
            section += (number or 1) * small_units[character]
            number = 0
    return total + section + number


def _fact_pattern(normalized_fact: str) -> str:
    """为事实短语中的数字加完整边界，防止 14 小时命中 4 小时。"""

    pieces: list[str] = []
    cursor = 0
    for number in _ARABIC_NUMBER.finditer(normalized_fact):
        pieces.append(re.escape(normalized_fact[cursor : number.start()]))
        pieces.append(rf"(?<![\d.\-+]){re.escape(number.group())}(?![\d.])")
        cursor = number.end()
    pieces.append(re.escape(normalized_fact[cursor:]))
    return "".join(pieces)


def _contains_coordinated_negative(text: str, fact: str) -> bool:
    """识别同一句中由一个否定词共同修饰的并列对象或动作。"""

    normalized_fact = _normalize_answer(fact)
    for prefix in _COORDINATED_NEGATIONS:
        if not normalized_fact.startswith(prefix) or len(normalized_fact) == len(prefix):
            continue
        tail_pattern = _fact_pattern(normalized_fact[len(prefix) :])
        pattern = rf"{re.escape(prefix)}.{{1,28}}(?:或|和|及|与){tail_pattern}"
        # 按句界和转折拆开，否定范围不能越过“但是”等后续断言。
        for clause in re.split(r"[，,。；;！？!?]|(?:但是|但|却|然而)", text):
            if re.search(pattern, _normalize_answer(clause)):
                return True
    return False


def _contains_fact(text: str, fact: str) -> bool:
    """匹配完整事实及其共享否定的并列说法，并保持数字边界。"""

    normalized_fact = _normalize_answer(fact)
    if not normalized_fact:
        return False
    if re.search(_fact_pattern(normalized_fact), _normalize_answer(text)):
        return True
    return _contains_coordinated_negative(text, fact)


def _has_contradictory_fact(
    answer: str, fact_groups: list[list[str]]
) -> bool:
    """识别“并非未列出”等反转句式，避免只因命中词面就判为正确。"""

    normalized_answer = _normalize_answer(answer)
    for group in fact_groups:
        for fact in group:
            normalized_fact = _normalize_answer(fact)
            if not normalized_fact:
                continue
            pattern = _CONTRADICTORY_PREFIX.pattern + r".{0,8}" + re.escape(normalized_fact)
            if re.search(pattern, normalized_answer) or re.search(
                _fact_pattern(normalized_fact) + _CONTRADICTORY_SUFFIX.pattern, normalized_answer
            ):
                return True
            for negative, positives in _NEGATIVE_TO_POSITIVE.items():
                if negative not in normalized_fact:
                    continue
                for positive in positives:
                    opposite = normalized_fact.replace(negative, positive, 1)
                    for clause in _CLAIM_SEPARATOR.split(_CITATION_MARKER.sub("", answer)):
                        normalized_clause = _normalize_answer(clause)
                        if _contains_fact(clause, fact):
                            continue
                        for match in re.finditer(_fact_pattern(opposite), normalized_clause):
                            prefix = normalized_clause[max(0, match.start() - 8) : match.start()]
                            if not re.search(r"(?:不|未|没有|不会|不能|并非)$", prefix):
                                return True
    return False


def _unverified_answer_claims(
    answer: str,
    fact_groups: list[list[str]],
    cited_texts: list[str],
    query: str = "",
) -> list[str]:
    """列出无法用预期事实或引用原文自动核对的子句，留待人工复核。"""

    normalized_sources = [_normalize_answer(text) for text in cited_texts]
    clauses = _CLAIM_SEPARATOR.split(_CITATION_MARKER.sub("", answer))
    unverified: list[str] = []
    for clause in clauses:
        normalized_clause = _normalize_answer(clause)
        if len(normalized_clause) <= 1:
            continue
        for lead in _NON_FACTUAL_LEADS:
            normalized_clause = normalized_clause.removeprefix(_normalize_answer(lead))
        if len(normalized_clause) <= 1:
            continue
        if _INCOMPLETE_CLAIM.fullmatch(normalized_clause):
            # “没有写”等无对象片段不是完整事实；后续完整分句仍需单独核验。
            continue
        if _supported_short_confirmation(normalized_clause, query, fact_groups, cited_texts):
            continue
        if _supported_unit_correction(normalized_clause, query, normalized_sources):
            continue
        if normalized_clause.endswith(("时", "后")) and _source_supports_clause(
            normalized_clause[:-1], normalized_sources
        ):
            # “发现异常时，”是有来源的条件前缀，由后续子句承担实际结论。
            continue
        matches_expected_fact = _expected_fact_covers_clause(
            normalized_clause, fact_groups, normalized_sources
        )
        matches_source_text = _source_supports_clause(normalized_clause, normalized_sources)
        if not matches_expected_fact and not matches_source_text:
            unverified.append(clause.strip())
    return unverified


def _expected_fact_covers_clause(
    clause: str, fact_groups: list[list[str]], normalized_sources: list[str]
) -> bool:
    """核对预期事实之外的剩余文字，只允许有来源的主语或条件描述。"""

    remaining = clause
    supported_sources: list[str] = []
    for group in fact_groups:
        # 每组列出同一个事实的接受表达；答案与原文可以各用其中一种。
        group_sources = [
            source for source in normalized_sources
            if any(_contains_fact(source, alternative) for alternative in group)
        ]
        for fact in group:
            normalized_fact = _normalize_answer(fact)
            if not normalized_fact:
                continue
            pattern = _fact_pattern(normalized_fact)
            if re.search(pattern, remaining):
                remaining = re.sub(pattern, "|", remaining)
                supported_sources.extend(group_sources)
    if remaining == clause or not supported_sources:
        return False
    # 命中短语不能豁免其余断言；不能确定语义的余下说法交给人工复核。
    fragments = [fragment for fragment in remaining.split("|") if fragment]
    return all(
        not _RESIDUAL_ASSERTION.search(fragment)
        and _source_supports_clause(fragment, supported_sources)
        for fragment in fragments
    )


def _source_supports_clause(clause: str, normalized_sources: list[str]) -> bool:
    """核对引用原文中的子句，容许“进行”等不改变事实的轻动词。"""

    if not clause:
        return False
    # 肯定子串落在“不提供”等否定断言中时，不能视为原文支持。
    pattern = r"(?<![不未没])" + _fact_pattern(clause)
    if any(re.search(pattern, source) for source in normalized_sources):
        return True
    without_light_verb = clause.replace("进行", "")
    return len(without_light_verb) >= 4 and any(
        re.search(r"(?<![不未没])" + _fact_pattern(without_light_verb), source.replace("进行", ""))
        for source in normalized_sources
    )


def _supported_short_confirmation(
    clause: str,
    query: str,
    fact_groups: list[list[str]],
    cited_texts: list[str],
) -> bool:
    """仅在问句与已核对事实极性一致时接受独立的“可以/不可以”。"""

    if clause not in {"可以", "不可以", "不能", "不会"}:
        return False
    if not re.search(r"(?:可以|能|会).{0,20}(?:吗|？|\?)", query):
        return False
    negative_fact = any(
        any(marker in fact for marker in ("不", "仅供"))
        for group in fact_groups
        for fact in group
    )
    if clause != "可以":
        return negative_fact
    return not negative_fact and any("可" in text for text in cited_texts)


def _supported_unit_correction(
    clause: str, query: str, normalized_sources: list[str]
) -> bool:
    """引文明确按小时给出上限时，允许回答纠正提问中的“天”。"""

    return bool(
        clause in {"不是按天计算", "不按天计算"}
        and re.search(r"(?:几|多少)天", query)
        and any(re.search(r"最长\d+小时", source) for source in normalized_sources)
    )


_CITATION_MARKER = re.compile(r"\[\d+\]")


def _unsupported_numeric_claims(answer: str, cited_texts: list[str]) -> list[str]:
    """核对完整带单位数值及独立数字，避免把负号或单位拆成伪断言。"""

    without_labels = _CITATION_MARKER.sub("", answer)
    without_labels = re.sub(r"([+\-−－负正])\s+(?=[\d零〇一二两三四五六七八九十百千万亿])", r"\1", without_labels)
    unit_matches = list(_NUMERIC_UNIT_CLAIM.finditer(without_labels))
    unit_spans = [match.span() for match in unit_matches]
    claims = [match.group() for match in unit_matches]
    # 已归入完整带单位数值的子匹配不再单独核对，例如“负18”中的“18”及“千瓦”中的“千”。
    for match in re.finditer(_NUMERIC_VALUE, without_labels):
        if not any(start <= match.start() and match.end() <= end for start, end in unit_spans):
            claims.append(match.group())
    return list(dict.fromkeys(
        claim for claim in claims if not any(_contains_fact(source, claim) for source in cited_texts)
    ))


def _assess_citation_claims(
    answer: str, fact_groups: list[list[str]], citations: list[Any], query: str
) -> tuple[bool | None, list[dict[str, Any]]]:
    """逐句限定到实际编号对应的引文；明确错配失败，不确定关系留待复核。"""

    statements = cited_statements(answer)
    records: list[dict[str, Any]] = []
    for index, statement in enumerate(statements):
        normalized = _normalize_answer(statement.text)
        for lead in _NON_FACTUAL_LEADS:
            normalized = normalized.removeprefix(_normalize_answer(lead))
        if not normalized or _INCOMPLETE_CLAIM.fullmatch(normalized):
            continue
        ids = statement.citation_ids
        # 短确认语可与紧随的解释共用引用，完整事实不得默认借用下一句的编号。
        if not ids and index + 1 < len(statements):
            next_ids = statements[index + 1].citation_ids
            next_texts = [citations[label - 1].text for label in next_ids if 1 <= label <= len(citations)]
            if _supported_short_confirmation(normalized, query, fact_groups, next_texts):
                ids = next_ids
        texts = [citations[label - 1].text for label in ids if 1 <= label <= len(citations)]
        status: bool | None = True
        reason = "对应引文支持"
        if not ids:
            status, reason = None, "断言未标注引用或跨句引用范围不明确"
        elif len(texts) != len(ids):
            status, reason = False, "引用编号不存在"
        else:
            matched_groups = [
                group for group in fact_groups
                if any(_contains_fact(statement.text, fact) for fact in group)
            ]
            mismatched_fact = any(
                not any(_contains_fact(text, fact) for text in texts for fact in group)
                for group in matched_groups
            )
            unsupported_numbers = _unsupported_numeric_claims(statement.text, texts)
            if mismatched_fact or unsupported_numbers:
                status, reason = False, "断言事实或数值不受对应编号的原文支持"
            elif _unverified_answer_claims(statement.text, fact_groups, texts, query):
                status, reason = None, "对应引文不能自动确认完整断言"
        records.append({
            "statement": statement.text, "citation_ids": list(ids),
            "supported": status, "reason": reason,
        })
    if not records:
        return None, records
    if any(record["supported"] is False for record in records):
        return False, records
    if any(record["supported"] is None for record in records):
        return None, records
    return True, records


def assess_case(
    case: dict[str, Any], response: Any | None, *, error: str | None = None
) -> dict[str, Any]:
    """记录单题检索、作答、引用支持和拒答质量；执行错误必定不通过。"""

    outcome: dict[str, Any] = {
        "id": case["id"],
        "answerable": bool(case["answerable"]),
        "expected_evidence_hit": False,
        "expected_evidence_rank": None,
        "top1_score": None,
        "rejected": None,
        "normal_answer": False,
        "answer_correct": False if case["answerable"] else None,
        "citations_traceable": False,
        "citation_supports_expected_evidence": False,
        "manual_review_status": "执行错误" if error else "待核验",
        "quality_pass": False,
        "answer": "",
        "citations": [],
        "provided_chunk_count": 0,
    }
    if error or response is None:
        outcome["error"] = error or "MissingResponse"
        return outcome

    raw_expected_evidence = case.get("expected_evidence")
    expected_evidence = (
        [raw_expected_evidence]
        if isinstance(raw_expected_evidence, str)
        else list(raw_expected_evidence or [])
    )
    evidence_ranks = []
    for evidence in expected_evidence:
        matching_ranks = [
            result.candidate_rank or rank
            for rank, result in enumerate(response.results, start=1)
            if result.chunk.file_name == case["expected_source"]
            and evidence in result.chunk.content
        ]
        if matching_ranks:
            evidence_ranks.append(min(matching_ranks))
    # 多证据问题的排名取最末一条所需证据的排名，要求 Top-K 含有全部证据。
    expected_evidence_rank = (
        max(evidence_ranks)
        if expected_evidence and len(evidence_ranks) == len(expected_evidence)
        else None
    )
    citations = list(response.citations)
    result_chunks = {result.chunk.chunk_id: result.chunk for result in response.results}
    outcome.update(
        {
            # 即使 RAG 对存在性问题扩展候选集，也只把前五名计入 Top-5 指标。
            "expected_evidence_hit": (
                expected_evidence_rank is not None and expected_evidence_rank <= 5
            ),
            "expected_evidence_rank": expected_evidence_rank,
            "top1_score": round(response.results[0].score, 4) if response.results else None,
            "rejected": bool(response.rejected),
            "normal_answer": not response.rejected,
            "answer": response.answer,
            "provided_chunk_count": len(response.results),
            "citations": [
                {"label": label, "chunk_id": citation.chunk_id, "file_name": citation.file_name,
                 "page_number": citation.page_number, "text": citation.text}
                for label, citation in enumerate(citations, start=1)
            ],
        }
    )

    if case["answerable"]:
        answer = response.answer
        fact_groups = case.get("expected_answer_facts", [])
        facts_match = bool(fact_groups) and all(
            any(_contains_fact(answer, fact) for fact in group)
            for group in fact_groups
        )
        facts_supported_by_citations = bool(fact_groups) and all(
            any(
                _contains_fact(citation.text, fact)
                for citation in citations
                for fact in group
            )
            for group in fact_groups
        )
        contradictory_fact = _has_contradictory_fact(answer, fact_groups)
        citations_traceable = bool(citations) and all(
            citation.chunk_id in result_chunks
            and citation.file_name == result_chunks[citation.chunk_id].file_name
            and citation.page_number == result_chunks[citation.chunk_id].page_number
            and citation.text == result_chunks[citation.chunk_id].content
            for citation in citations
        )
        # 固定题集声明了唯一预期文档，其他文档的同名条款不能混入该题回答。
        unexpected_sources = sorted({
            citation.file_name for citation in citations
            if citation.file_name != case["expected_source"]
        })
        citation_source_scope_valid = bool(citations) and not unexpected_sources
        citation_supports_evidence = bool(expected_evidence) and all(
            any(
                citation.file_name == case["expected_source"]
                and evidence in citation.text
                for citation in citations
            )
            for evidence in expected_evidence
        )
        cited_texts = [citation.text for citation in citations]
        unsupported_numbers = _unsupported_numeric_claims(answer, cited_texts)
        unverified_claims = _unverified_answer_claims(
            answer, fact_groups, cited_texts, case.get("query", "")
        )
        citation_claim_support, citation_claims = _assess_citation_claims(
            answer, fact_groups, citations, case.get("query", "")
        )
        if contradictory_fact or unsupported_numbers:
            answer_correct: bool | None = False
        elif not facts_match or unverified_claims:
            # 规则无法确认答案事实或补充陈述时，须人工复核而不能直接通过。
            answer_correct = None
        else:
            answer_correct = True

        citation_basis_valid = (
            citations_traceable
            and citation_source_scope_valid
            and citation_supports_evidence
            and facts_supported_by_citations
        )
        if (response.rejected or not citation_basis_valid or answer_correct is False
                or citation_claim_support is False):
            manual_review_status = "自动核验失败"
        elif answer_correct is None or citation_claim_support is None:
            manual_review_status = "待人工复核"
        else:
            manual_review_status = "自动核验通过"
        quality_pass = (
            not response.rejected
            and answer_correct is True
            and facts_supported_by_citations
            and citations_traceable
            and citation_source_scope_valid
            and citation_supports_evidence
            and citation_claim_support is True
            and manual_review_status == "自动核验通过"
        )
        outcome.update(
            {
                "expected_fact_phrases_match": facts_match,
                "expected_facts_supported_by_citations": facts_supported_by_citations,
                "contradictory_fact": contradictory_fact,
                "unsupported_numeric_claims": unsupported_numbers,
                "unverified_claims": unverified_claims,
                "answer_correct": answer_correct,
                "manual_review_status": manual_review_status,
                "citations_traceable": citations_traceable,
                "citation_source_scope_valid": citation_source_scope_valid,
                "unexpected_citation_sources": unexpected_sources,
                "citation_supports_expected_evidence": citation_supports_evidence,
                "citation_claims_supported": citation_claim_support,
                "citation_claims": citation_claims,
                "quality_pass": quality_pass,
            }
        )
    else:
        # 无答案项要求拒答状态、明确拒答文本和引用状态彼此一致。
        refusal_text = has_explicit_refusal(response.answer)
        answer_has_citation = bool(_CITATION_MARKER.search(response.answer))
        outcome["refusal_text_present"] = refusal_text
        outcome["answer_has_citation"] = answer_has_citation
        outcome["quality_pass"] = bool(
            response.rejected
            and refusal_text
            and not citations
            and not answer_has_citation
        )
        outcome["manual_review_status"] = (
            "自动核验通过" if outcome["quality_pass"] else "自动核验失败"
        )
    return outcome


