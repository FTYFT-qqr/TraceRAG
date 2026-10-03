"""准备原文短句并校验模型选择的编号，不依赖检索策略或模型 SDK。"""

from __future__ import annotations

import json
import re


REFUSAL_ANSWER = "根据当前知识库，我无法确认这个问题。"
_SENTENCE_BOUNDARY = re.compile(r"[。；;！？!?\n]+|\.(?!\d)")
_PASSAGE_BOUNDARY = re.compile(r"[。！？!?\n]+|\.(?!\d)")


def evidence_passages(evidence: dict[int, str]) -> dict[str, tuple[int, str]]:
    """按完整句界拆分证据，保存短句编号到来源编号及原文的稳定映射。"""

    passages: dict[str, tuple[int, str]] = {}
    for source_id, text in evidence.items():
        # 分号连接的前置手续和确认条件同属一个完整句，不能拆成可单独选择的残片。
        for index, part in enumerate(_PASSAGE_BOUNDARY.split(text), start=1):
            quote = part.strip()
            if quote:
                passages[f"{source_id}:{index}"] = (source_id, quote)
    return passages


def render_grounded_answer(content: str, passages: dict[str, tuple[int, str]]) -> str:
    """验证 JSON、拒答状态与全部短句编号，再从原文生成逐句引用。"""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Chat API must return a grounded answer JSON object.") from exc
    if not isinstance(payload, dict) or set(payload) != {"refused", "passage_ids"}:
        raise ValueError("Grounded answer requires only refused and passage_ids fields.")
    refused, ids = payload["refused"], payload["passage_ids"]
    if not isinstance(refused, bool) or not isinstance(ids, list):
        raise ValueError("Grounded answer has invalid refusal or passage ID types.")
    if refused:
        if ids:
            raise ValueError("A refusal cannot contain evidence passages.")
        return REFUSAL_ANSWER
    if not ids:
        raise ValueError("A non-refusal must select evidence passages.")
    if any(not isinstance(label, str) or label not in passages for label in ids):
        raise ValueError("Grounded answer references an invalid passage ID.")

    sentences: list[str] = []
    for label in dict.fromkeys(ids):
        source_id, quote = passages[label]
        # 原文中的方括号数字保留为全角，避免被 RAG 误当作本次生成的引用编号。
        quote = re.sub(r"\[(\d+)\]", r"［\1］", quote)
        for sentence in _SENTENCE_BOUNDARY.split(quote):
            if sentence.strip():
                sentences.append(f"{sentence.strip()}[{source_id}]。")
    return "\n".join(dict.fromkeys(sentences))
