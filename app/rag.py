"""串联向量检索、证据约束回答、来源引用和低置信度拒答。"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Protocol

from app.models import Citation, QueryResponse, RetrievalResult
from app.grounded_answer import REFUSAL_ANSWER
from app.relevance import RelevancePolicy


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
# 只把明确表达“无法判断/回答”的措辞视为拒答，不把有证据的否定事实当成拒答。
_REFUSAL_PATTERNS = (
    re.compile(r"(?:无法|不能|没法|未能|没有办法)(?:从.{0,24})?(?:确认|确定|回答|判断|得知)"),
    re.compile(r"\b(?:cannot|can't|unable to)\s+(?:confirm|determine|answer|verify)\b", re.IGNORECASE),
)
_MISSING_DETAIL_PATTERNS = (
    re.compile(r"(?:未列出|没有列出|未提供|没有提供|未说明|没有说明|未提及|没有提及|未记载|没有记载)"),
    re.compile(r"\b(?:not|does not|doesn't)\s+(?:specify|list|state|mention|provide)\b", re.IGNORECASE),
)
_INFORMATION_PRESENCE_QUESTION = re.compile(
    r"(?:是否|有没有|有无|有沒有).{0,24}(?:列出|提供|说明|提及|包含|记载|写明|规定|标明|注明)"
    r"|(?:列出|提供|说明|提及|包含|记载|写明|规定|标明|注明|写).{0,8}(?:吗|没有|了吗|与否)"
    r"|(?:文档|文件|资料|本规定|规定|手册|政策|制度).{0,16}"
    r"(?:有.{0,4}(?:写|列出|提供|说明|提及|记载|包含)|没有.{0,4}(?:写|列出|提供|说明|提及|记载|包含)|有没有|有无|是否)"
    r".{0,18}(?:吗|没有|与否)?"
    r"|\b(?:does|did)\s+(?:the\s+)?(?:document|manual|policy|rule)\s+"
    r"(?:list|specify|state|mention|provide)\b",
    re.IGNORECASE,
)
_SPECIFIC_VALUE_QUESTION = re.compile(
    r"(?:具体|确切|实际)?(?:金额|数额|费用|赔偿金|价格)?"
    r"(?:具体|确切|实际)?(?:是)?(?:多少(?:元|块|钱)?|几(?:元|块|天|小时)|"
    r"多少钱|什么(?:金额|数额))"
)


def has_explicit_refusal(answer: str) -> bool:
    """判断答案是否明确表示无法确认或回答。"""

    return REFUSAL_ANSWER in answer or any(
        pattern.search(answer) for pattern in _REFUSAL_PATTERNS
    )


def _is_refusal(query: str, answer: str) -> bool:
    """根据明确拒答语和问题意图区分缺失信息与可引用的否定事实。"""

    # 生成内容明确表示无法回答时优先拒答，不能被后续的否定事实例外覆盖。
    if has_explicit_refusal(answer):
        return True
    # 混合问题若同时询问“是否记载”和具体数值，可引用回答“未记载”，但不得编造数值。
    asks_whether_listed = bool(_INFORMATION_PRESENCE_QUESTION.search(query))
    states_missing_detail = any(
        pattern.search(answer) for pattern in _MISSING_DETAIL_PATTERNS
    )
    if asks_whether_listed and states_missing_detail:
        return False
    # “文件未列出某项”是可引用事实；仅在单独索取缺失具体值时判为拒答。
    return states_missing_detail and bool(_SPECIFIC_VALUE_QUESTION.search(query))


_PRESENCE_QUERY_FILLERS = re.compile(
    r"(?:请问|麻烦|文档|文件|资料|本规定|规定|手册|政策|制度|其中|里面|是否|有没有|有无|有没|"
    r"可以|能否|会不会|写明|列出|提供|说明|提及|记载|包含|注明|标明|写|有|吗|呢|具体|确切|实际|"
    r"到底|多少|几|什么|多少钱|多少元)"
)


def _has_topic_overlap(query: str, content: str) -> bool:
    """判断片段是否包含问题主题，用于挑选扩展候选中的否定证据。"""

    topic = _PRESENCE_QUERY_FILLERS.sub("", query).casefold()
    topic = re.sub(r"[^\w\u4e00-\u9fff]", "", topic)
    normalized_content = re.sub(r"[^\w\u4e00-\u9fff]", "", content).casefold()
    if not topic:
        return False
    if topic in normalized_content:
        return True

    # 中文问题按相邻双字比较主题覆盖率，英文则保留完整词，避免仅凭常见问句词提升片段。
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", topic)
    chinese_terms = {
        run[index : index + 2]
        for run in chinese_runs
        for index in range(len(run) - 1)
    }
    latin_terms = set(re.findall(r"[a-z0-9]{2,}", topic))
    terms = chinese_terms | latin_terms
    return bool(terms) and sum(term in normalized_content for term in terms) / len(terms) >= 0.5


def _select_results(
    query: str, candidates: list[RetrievalResult], top_k: int
) -> list[RetrievalResult]:
    """在最多 Top-K 段的约束内，优先保留扩展候选中的相关否定证据。"""

    # 假检索器没有记录原始名次时按返回顺序补齐，生产检索器会提供真实候选名次。
    ranked = [
        result
        if result.candidate_rank is not None
        else replace(result, candidate_rank=index)
        for index, result in enumerate(candidates, start=1)
    ]
    selected = ranked[:top_k]
    if not _INFORMATION_PRESENCE_QUESTION.search(query) or not selected:
        return selected

    # 只提升同主题且明确包含“未列出”等措辞的片段，避免无关缺项段抢占参考位。
    negative_evidence = next(
        (
            result
            for result in ranked[top_k:]
            if any(pattern.search(result.chunk.content) for pattern in _MISSING_DETAIL_PATTERNS)
            and _has_topic_overlap(query, result.chunk.content)
        ),
        None,
    )
    if negative_evidence is not None:
        selected[-1] = negative_evidence
        selected.sort(key=lambda result: result.candidate_rank or 0)
    return selected


class Retriever(Protocol):
    """定义 RAG 流程所需的最小检索接口。"""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """按相关度返回带来源元数据的检索候选。"""
        ...


class AnswerGenerator(Protocol):
    """定义根据问题和已排序证据生成回答的接口。"""

    def answer(self, query: str, results: list[RetrievalResult]) -> str:
        """根据问题和候选证据生成带编号引用的回答文本。"""
        ...


class RAGService:
    """仅在检索分数达标且回答引用有效来源时返回成功结果。"""

    def __init__(
        self,
        retriever: Retriever,
        chat_client: AnswerGenerator,
        *,
        reject_threshold: float = 0.25,
        relevance_policy: RelevancePolicy | None = None,
    ) -> None:
        """组合检索器和回答器，并固定低置信度拒答阈值。"""
        if not -1.0 <= reject_threshold <= 1.0:
            raise ValueError("reject_threshold must be between -1 and 1.")
        self._retriever = retriever
        self._chat_client = chat_client
        self.reject_threshold = reject_threshold
        self.relevance_policy = relevance_policy or RelevancePolicy(vector_min_score=reject_threshold)

    def query(self, query: str, top_k: int = 5) -> QueryResponse:
        """相关度不足、引用无效或生成拒答语时统一返回标准拒答。"""

        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # 是否记载类问题需要扩大候选范围，避免同义问法漏掉明确的否定证据段。
        retrieval_top_k = (
            top_k + 5
            if _INFORMATION_PRESENCE_QUESTION.search(query)
            else top_k
        )
        candidates = self._retriever.retrieve(query, top_k=retrieval_top_k)
        # 模型上下文、接口调试列表和引用映射共用同一份不超过用户上限的证据。
        results = _select_results(query, candidates, top_k)
        if not self.relevance_policy.allows(results):
            return QueryResponse(REFUSAL_ANSWER, True, (), tuple(results))

        generated = self._chat_client.answer(query, results)
        if not isinstance(generated, str) or not generated.strip():
            return QueryResponse(REFUSAL_ANSWER, True, (), tuple(results))
        # 明确拒答优先于引用编号；缺失信息只有在用户询问具体值时才算拒答。
        if _is_refusal(query, generated):
            return QueryResponse(REFUSAL_ANSWER, True, (), tuple(results))

        source_ids = {index for index, _ in enumerate(results, start=1)}
        cited_ids = [int(match.group(1)) for match in _CITATION_PATTERN.finditer(generated)]
        # 编号只验证“是否对应本次检索结果”，并不能自动证明事实受证据支持。
        if not cited_ids or any(source_id not in source_ids for source_id in cited_ids):
            return QueryResponse(REFUSAL_ANSWER, True, (), tuple(results))
        used_ids = list(dict.fromkeys(cited_ids))

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
