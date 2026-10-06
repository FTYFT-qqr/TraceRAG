"""验证 006 语料入口、PDF 页码和分阶段报告。"""

from pathlib import Path

import pytest

from app.config import Settings
from app.models import Chunk, RetrievalResult
from scripts.evaluate_v02 import (
    BenchmarkSpec,
    _find_expected_rank,
    _load_pages,
    load_cases,
    run_benchmark,
    validate_cases_against_corpus,
)


BUSINESS_DIR = Path("samples/business/006")


def test_006_development_and_acceptance_sources_are_valid() -> None:
    """确认开发集可少于 50 题，正式集严格采用 40 答案加 10 拒答。"""

    development = load_cases(
        BUSINESS_DIR / "questions_development.json", dataset_mode="development"
    )
    acceptance = load_cases(
        BUSINESS_DIR / "questions_acceptance.json", dataset_mode="acceptance"
    )
    assert len(development) == 12
    assert len(acceptance) == 50
    assert sum(case["answerable"] for case in acceptance) == 40
    validate_cases_against_corpus(development, BUSINESS_DIR)
    validate_cases_against_corpus(acceptance, BUSINESS_DIR)


def test_006_pdf_evidence_rejects_wrong_page() -> None:
    """确认 PDF 原文即使正确，页码错误也不能通过校验。"""

    cases = load_cases(
        BUSINESS_DIR / "questions_acceptance.json", dataset_mode="acceptance"
    )
    pdf_case = next(
        case for case in cases
        if case["answerable"] and case["expected_source"].lower().endswith(".pdf")
    )
    altered = {**pdf_case, "expected_page_number": pdf_case["expected_page_number"] + 100}
    with pytest.raises(ValueError, match="Expected evidence is absent"):
        validate_cases_against_corpus([altered], BUSINESS_DIR)


def test_006_loader_rejects_duplicate_names_and_blank_text(tmp_path: Path) -> None:
    """确认来源文件名唯一，空白文件不能伪装成可用语料。"""

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "规则.txt").write_text("有效条款", encoding="utf-8")
    (second / "规则.txt").write_text("另一条款", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate evaluation source"):
        _load_pages(tmp_path)
    (second / "规则.txt").unlink()
    (second / "空白.md").write_text("  \n  ", encoding="utf-8")
    with pytest.raises(ValueError, match="no usable text"):
        _load_pages(tmp_path)


def test_006_pdf_rank_requires_matching_page() -> None:
    """确认错误 PDF 页上的相同文字不会虚报 Top-5 命中。"""

    case = {
        "answerable": True,
        "expected_source": "政策.pdf",
        "expected_evidence": "共同文字",
        "expected_page_number": 2,
    }
    wrong_page = Chunk("one", "doc", "共同文字", "政策.pdf", 1, 0)
    right_page = Chunk("two", "doc", "共同文字", "政策.pdf", 2, 1)
    assert _find_expected_rank(case, [RetrievalResult(wrong_page, 0.8)]) is None
    assert _find_expected_rank(case, [
        RetrievalResult(wrong_page, 0.8), RetrievalResult(right_page, 0.7)
    ]) == 2


def test_006_bm25_report_separates_initialization_and_online_probe(tmp_path: Path) -> None:
    """确认纯 BM25 可独立运行，并且在线单题延迟没有混入离线批量耗时。"""

    (tmp_path / "规则.txt").write_text("办理期限是三天。", encoding="utf-8")
    case = {
        "id": "T01", "query": "办理期限是多少？", "answerable": True,
        "expected_source": "规则.txt", "expected_evidence": "办理期限是三天。",
        "expected_answer_facts": ["三天"], "allow_citations": True,
        "refusal_expected": False, "manual_review_condition": "未知说法", "tags": ["numeric"],
    }
    result = run_benchmark(
        Settings(), [case], BenchmarkSpec("bm25-probe", strategy="bm25"),
        quality=False, corpus_dir=tmp_path,
    )
    performance = result["performance"]
    assert performance["stages"]["initialization"]["status"] == "已采集"
    assert performance["stages"]["model_load"]["status"] == "未采集"
    assert performance["latency"]["offline_retrieval"]["sample_count"] == 1
    assert performance["latency"]["online_retrieval"]["sample_count"] == 1
    assert performance["latency"]["online_total"]["status"] == "未采集"
    assert result["cases"][0]["chat_called"] is False
