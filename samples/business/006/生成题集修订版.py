"""从冻结的 006 验收题集生成仅修正 B006-A1-17 问法的 v2。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD_NAME = "questions_acceptance.json"
NEW_NAME = "questions_acceptance_v2.json"
NEW_QUERY = "客户收件后多长时间内可申请哪项复核？"


def _sha256(path: Path) -> str:
    """按原始字节计算冻结输入指纹。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """只读校验 v1，再新增 v2 题集和独立元数据。"""

    old_metadata = json.loads((ROOT / "语料元数据.json").read_text(encoding="utf-8"))
    old_path = ROOT / OLD_NAME
    old_hash = _sha256(old_path)
    if old_hash != old_metadata["sha256"][OLD_NAME]:
        raise ValueError("冻结的 v1 题集与原元数据指纹不一致，停止生成。")
    cases = json.loads(old_path.read_text(encoding="utf-8"))
    target = [case for case in cases if case["id"] == "B006-A1-17"]
    if len(target) != 1:
        raise ValueError("目标题号不是唯一。")
    original_query = target[0]["query"]
    if original_query != "客户对尺寸复测结果有异议时可以做什么？":
        raise ValueError("原问法已变化，须重新人工复核修订理由。")
    if target[0]["expected_evidence"] != "客户可在收件后2个工作日内申请尺寸复核。":
        raise ValueError("来源证据已变化，不能仅修订问法。")
    if target[0]["expected_answer_facts"] != [["2个工作日内"], ["申请尺寸复核"]]:
        raise ValueError("预期事实已变化，不能仅修订问法。")
    target[0]["query"] = NEW_QUERY

    new_path = ROOT / NEW_NAME
    new_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    new_hash = _sha256(new_path)
    metadata = {
        "dataset": "TraceRAG-006-acceptance-v2",
        "revision_date": "2026-10-05",
        "revision_reason": "B006-A1-17 的 v1 问法把来源未声明的复测结果异议流程当作前提。",
        "source_type": old_metadata["source_type"],
        "corpus_files": old_metadata["corpus_files"],
        "development_questions": old_metadata["development_questions"],
        "acceptance_questions": NEW_NAME,
        "answerable_count": 40,
        "unanswerable_count": 10,
        "source_v1_question_sha256": old_hash,
        "sha256": {
            **{name: old_metadata["sha256"][name] for name in old_metadata["corpus_files"]},
            old_metadata["development_questions"]: old_metadata["sha256"][old_metadata["development_questions"]],
            NEW_NAME: new_hash,
        },
        "changed_case_id": "B006-A1-17",
        "changed_field": "query",
        "original_query": original_query,
        "revised_query": NEW_QUERY,
        "expected_facts_unchanged": True,
    }
    (ROOT / "语料元数据_v2.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"v1={old_hash}\nv2={new_hash}")


if __name__ == "__main__":
    main()
