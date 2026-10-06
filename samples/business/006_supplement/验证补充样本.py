"""用正式 Loader 与评测 schema 验证 006 整改后补充回归样本。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))

from app.chunking import chunk_pages  # noqa: E402
from app.document_loader import load_document  # noqa: E402
from scripts.evaluate_v02 import load_cases, validate_cases_against_corpus  # noqa: E402


def _evidence_set(cases: list[dict]) -> set[str]:
    """收集有答案题的完整证据原文，供独立性核对。"""

    result = set()
    for case in cases:
        raw = case.get("expected_evidence")
        result.update([raw] if isinstance(raw, str) else raw or [])
    return result


def main() -> None:
    """验证指纹、原文事实、数值干扰、缺失信息和旧题集隔离。"""

    metadata = json.loads((ROOT / "语料元数据.json").read_text(encoding="utf-8"))
    assert metadata["source_type"] == "原创模拟业务语料"
    assert metadata["not_full_independent_acceptance"] is True
    assert metadata["answerable_count"] == 12 and metadata["unanswerable_count"] == 3
    for name, expected in metadata["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, (
            f"补充输入指纹变化：{name}"
        )

    source_name = metadata["corpus_files"][0]
    source = (ROOT / source_name).read_bytes()
    pages = load_document(source_name, source)
    assert len(pages) == 1 and pages[0].content.strip()
    text = pages[0].content
    chunks = chunk_pages(pages, chunk_size=700, overlap=100)

    cases = load_cases(ROOT / metadata["questions"], dataset_mode="development")
    validate_cases_against_corpus(cases, ROOT)
    assert len(cases) == 15 and len({case["query"] for case in cases}) == 15
    assert sum(case["answerable"] for case in cases) == 12
    assert sum(not case["answerable"] for case in cases) == 3

    numeric_distractors = 0
    for case in cases:
        assert case["expected_source"] == source_name
        if not case["answerable"]:
            assert case["expected_evidence"] is None
            assert case["expected_answer_facts"] == []
            assert case["refusal_expected"] and not case["allow_citations"]
            continue
        evidence = case["expected_evidence"]
        assert isinstance(evidence, str) and evidence in text
        assert any(evidence in chunk.content for chunk in chunks), (
            f"证据被切分边界截断：{case['id']}"
        )
        for group in case["expected_answer_facts"]:
            assert isinstance(group, list) and group
            assert any(fact in evidence for fact in group), (
                f"预期事实未见于证据：{case['id']} {group}"
            )
        if len(set(re.findall(r"\d+(?:\.\d+)?", evidence))) >= 2:
            numeric_distractors += 1
    assert numeric_distractors >= 6, "真正含两个不同数值的干扰条款不足"
    assert "负18摄氏度" in text and "负12摄氏度" in text
    assert "1.5千瓦" in text and "0.8千瓦" in text
    assert all(absent not in text for absent in ("制造商名称", "经纬度坐标", "每月电费"))

    original_root = ROOT.parent / "006"
    original_cases = json.loads((original_root / "questions_acceptance.json").read_text(encoding="utf-8"))
    assert not {case["query"] for case in cases}.intersection(
        case["query"] for case in original_cases
    ), "补充题目与原 50 题重合"
    assert not _evidence_set(cases).intersection(_evidence_set(original_cases)), (
        "补充证据与原 50 题重合"
    )
    assert source_name not in {case["expected_source"] for case in original_cases}

    print(json.dumps({
        "documents": 1,
        "answerable": 12,
        "unanswerable": 3,
        "numeric_distractor_cases": numeric_distractors,
        "negative_temperature": True,
        "chinese_units": ["摄氏度", "千瓦"],
        "original_50_query_overlap": 0,
        "original_50_evidence_overlap": 0,
        "sha256": metadata["sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
