"""使用正式文档 Loader 验证 006 语料、页码、事实和冻结指纹。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))

from app.document_loader import load_document  # noqa: E402
from app.chunking import chunk_pages  # noqa: E402


def _evidence_list(case: dict) -> list[str]:
    """将单条和多条预期证据统一成列表。"""

    raw = case["expected_evidence"]
    return [raw] if isinstance(raw, str) else list(raw or [])


def main() -> None:
    """逐项核对冻结文件、文档文本层和题集事实，失败即抛出错误。"""

    metadata = json.loads((ROOT / "语料元数据.json").read_text(encoding="utf-8"))
    assert metadata["source_type"] == "原创模拟业务语料"
    for name, expected_hash in metadata["sha256"].items():
        actual_hash = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"文件指纹变化：{name}"

    documents = {}
    chunks_by_source = {}
    for name in metadata["corpus_files"]:
        pages = load_document(name, (ROOT / name).read_bytes())
        assert pages and all(page.content.strip() for page in pages), f"无可用文本：{name}"
        documents[name] = pages
        chunks_by_source[name] = chunk_pages(pages, chunk_size=700, overlap=100)
    assert {Path(name).suffix for name in documents} == {".txt", ".md", ".pdf"}
    pdf_name = next(name for name in documents if name.endswith(".pdf"))
    assert [page.page_number for page in documents[pdf_name]] == [1, 2, 3]

    dev = json.loads((ROOT / metadata["development_questions"]).read_text(encoding="utf-8"))
    acceptance = json.loads((ROOT / metadata["acceptance_questions"]).read_text(encoding="utf-8"))
    assert len(dev) == 12 and all(case["answerable"] for case in dev)
    assert len(acceptance) == 50
    assert sum(case["answerable"] for case in acceptance) == 40
    assert sum(not case["answerable"] for case in acceptance) == 10
    all_cases = dev + acceptance
    assert len({case["id"] for case in all_cases}) == len(all_cases)
    assert len({case["query"] for case in all_cases}) == len(all_cases)

    required = {
        "id", "query", "answerable", "expected_source", "expected_evidence",
        "expected_answer_facts", "allow_citations", "refusal_expected",
        "manual_review_condition", "tags",
    }
    dev_evidence, acceptance_evidence = set(), set()
    pdf_development_count = pdf_acceptance_count = multi_segment_count = 0
    for case in all_cases:
        assert required <= case.keys(), f"题目缺字段：{case['id']}"
        source = case["expected_source"]
        assert source in documents, f"来源不存在：{case['id']}"
        assert case["query"].strip() and case["tags"]
        if not case["answerable"]:
            assert case["expected_evidence"] is None and case["expected_answer_facts"] == []
            assert case["refusal_expected"] and not case["allow_citations"]
            assert "expected_page_number" not in case
            continue
        assert case["allow_citations"] and not case["refusal_expected"]
        evidence_items = _evidence_list(case)
        assert evidence_items and case["expected_answer_facts"]
        if len(evidence_items) > 1:
            multi_segment_count += 1
        if source == pdf_name:
            if case in dev:
                pdf_development_count += 1
            else:
                pdf_acceptance_count += 1
            page_number = case.get("expected_page_number")
            assert isinstance(page_number, int), f"PDF 页码缺失：{case['id']}"
            matching_pages = [page for page in documents[source] if page.page_number == page_number]
            assert len(matching_pages) == 1
        else:
            assert "expected_page_number" not in case
            matching_pages = documents[source]
        for evidence in evidence_items:
            assert any(evidence in page.content for page in matching_pages), (
                f"原文或 PDF 页码不匹配：{case['id']} {evidence}"
            )
            assert any(evidence in chunk.content for chunk in chunks_by_source[source]), (
                f"预期原文被默认切分边界截断：{case['id']} {evidence}"
            )
            (dev_evidence if case in dev else acceptance_evidence).add((source, evidence))
        if len(evidence_items) > 1:
            assert not any(
                all(evidence in chunk.content for evidence in evidence_items)
                for chunk in chunks_by_source[source]
            ), f"多片段题的证据落入同一默认 Chunk：{case['id']}"
        for group in case["expected_answer_facts"]:
            assert isinstance(group, list) and group and all(isinstance(fact, str) for fact in group)
            assert any(
                alternative in evidence for alternative in group for evidence in evidence_items
            ), f"预期事实不在原文证据中：{case['id']} {group}"
    assert not dev_evidence.intersection(acceptance_evidence), "开发与验收共用证据"

    print(json.dumps({
        "documents": len(documents),
        "pdf_pages": len(documents[pdf_name]),
        "development_cases": len(dev),
        "acceptance_answerable": 40,
        "acceptance_unanswerable": 10,
        "pdf_development_answerable_cases": pdf_development_count,
        "pdf_acceptance_answerable_cases": pdf_acceptance_count,
        "multi_segment_cases": multi_segment_count,
        "shared_development_acceptance_evidence": 0,
        "sha256": metadata["sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
