"""解析答案断言与编号引用的对应关系，不依赖模型或检索实现。"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class CitedStatement:
    """记录一句断言及紧邻它的引用编号；未标注引用时编号为空。"""

    text: str
    citation_ids: tuple[int, ...] = ()


_STATEMENT_TOKEN = re.compile(
    r"(?P<citation>(?:\[\d+\]\s*)+)|"
    r"(?P<boundary>[。！？!?；;\n]+|\.(?!\d))"
)


def cited_statements(answer: str) -> list[CitedStatement]:
    """关联句末或句号后的引用，并保留未引用断言，避免跨句借用证据。"""

    statements: list[CitedStatement] = []
    cursor = 0
    for token in _STATEMENT_TOKEN.finditer(answer):
        text = answer[cursor : token.start()].strip(" \t\r，,")
        if token.lastgroup == "citation":
            ids = tuple(dict.fromkeys(int(value) for value in re.findall(r"\[(\d+)\]", token.group())))
            if text:
                statements.append(CitedStatement(text, ids))
            elif statements:
                previous = statements[-1]
                statements[-1] = replace(
                    previous, citation_ids=tuple(dict.fromkeys((*previous.citation_ids, *ids)))
                )
            else:
                # 前置引用没有明确对应的断言，交给评分器判断为不确定。
                statements.append(CitedStatement("", ids))
        elif text:
            statements.append(CitedStatement(text))
        cursor = token.end()
    tail = answer[cursor:].strip(" \t\r，,")
    if tail:
        statements.append(CitedStatement(tail))
    return statements
