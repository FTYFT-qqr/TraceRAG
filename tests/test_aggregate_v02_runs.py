"""验证跨策略汇总会严格匹配输入并保留真实失败。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.aggregate_v02_runs import aggregate_runs, markdown_report, write_aggregate


def _sha256(path: Path) -> str:
    """为测试题集和语料计算真实字节指纹。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_reports(tmp_path: Path) -> tuple[Path, list[Path], list[dict]]:
    """创建三份不同策略但相同输入配置的最小完整报告。"""

    corpus = tmp_path / "语料"
    corpus.mkdir()
    source = corpus / "规则.txt"
    source.write_text("办理期限是三天。", encoding="utf-8")
    questions = corpus / "questions.json"
    cases = [
        {
            "id": "A", "query": "办理期限？", "answerable": True,
            "expected_source": source.name, "expected_evidence": "办理期限是三天。",
            "expected_answer_facts": ["三天"], "allow_citations": True,
            "refusal_expected": False, "manual_review_condition": "额外说法", "tags": ["direct"],
        },
        {
            "id": "R", "query": "费用？", "answerable": False,
            "expected_source": source.name, "expected_evidence": None,
            "expected_answer_facts": [], "allow_citations": False,
            "refusal_expected": True, "manual_review_condition": "额外说法", "tags": ["refusal"],
        },
    ]
    questions.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    settings = {
        "embedding_identity": {"provider": "local", "model": "same", "model_fingerprint": "hash"},
        "implementation_sha256": {
            name: "same-hash" for name in (
                "scripts/evaluate_v02.py", "scripts/evaluation_performance.py",
                "app/document_loader.py", "app/rag.py", "app/evaluation.py",
            )
        },
        "chat_model": "model", "reject_threshold": 0.25,
    }
    reports = []
    paths = []
    for strategy in ("vector", "bm25", "hybrid"):
        report = {
            "schema_version": 1,
            "spec": {
                "name": strategy, "strategy": strategy, "chunk_size": 700,
                "overlap": 100, "rrf_k": 60, "embedding_batch_size": 1,
                "bm25_min_score": 0.0,
            },
            "inputs": {
                "questions_sha256": _sha256(questions),
                "corpus": {source.name: _sha256(source)},
                "corpus_dir": corpus.resolve().as_posix(),
            },
            "settings": copy.deepcopy(settings),
            "relevance_policy": {"vector_min_score": 0.25, "bm25_min_score": 0.0},
            "metrics": {
                "answerable_count": 1, "refusal_count": 1,
                "quality_executed": True, "quality_pass": 1, "refusal_pass": 1,
                "top1_hits": 1, "top3_hits": 1, "top5_hits": 1,
                "citation_traceable": 1, "citation_original_support": 1,
                "manual_review_count": 0, "chat_request_count": 1,
                "acceptance_pass": False, "errors": 0,
            },
            "cases": [
                {
                    "id": "A", "expected_evidence_rank": 1, "hit_top5": True,
                    "quality_pass": True, "chat_called": True, "chat_request_count": 1,
                    "manual_review_status": "自动核验通过",
                    "ranking": [{"rank": 1, "file_name": source.name, "page_number": None,
                                 "chunk_id": "one", "text": "办理期限是三天。"}],
                },
                {
                    "id": "R", "expected_evidence_rank": None, "hit_top5": False,
                    "quality_pass": True, "chat_called": False, "chat_request_count": 0,
                    "manual_review_status": "自动核验通过", "ranking": [],
                },
            ],
            "performance": {
                "total_seconds": 3.0,
                "hardware": {"os": "Windows"},
                "stages": {
                    "initialization": {"status": "已采集", "total_seconds": 1.0},
                    "model_load": {"status": "未采集", "total_seconds": None},
                },
                "latency": {
                    "online_retrieval": {
                        "status": "已采集", "sample_count": 2, "first_seconds": 0.1,
                        "steady_sample_count": 1, "steady_p50_seconds": 0.2,
                        "steady_p95_seconds": 0.2,
                    },
                    "chat": {
                        "status": "已采集", "sample_count": 1, "first_seconds": 0.5,
                        "steady_sample_count": 0, "steady_p50_seconds": None,
                        "steady_p95_seconds": None,
                    },
                },
                "memory": {"process_peak_rss_bytes": {"status": "已采集", "value": 104857600}},
                "model_calls": {
                    "chat": {"status": "已采集", "request_count": 1,
                             "failure_count": 0, "retry_count": None, "total_tokens": 10},
                    "embedding": {"status": "未采集", "request_count": None},
                },
            },
        }
        path = tmp_path / f"{strategy}.json"
        path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        reports.append(report)
        paths.append(path)
    return questions, paths, reports


def test_aggregate_writes_cross_run_quality_performance_and_failures(tmp_path: Path) -> None:
    """确认三策略汇总含质量、性能和失败类别，并保留原始报告。"""

    questions, paths, reports = _fixture_reports(tmp_path)
    reports[2]["cases"][0]["quality_pass"] = False
    reports[2]["metrics"]["quality_pass"] = 0
    paths[2].write_text(json.dumps(reports[2], ensure_ascii=False), encoding="utf-8")
    originals = [_sha256(path) for path in paths]
    aggregate = aggregate_runs(questions, paths)
    json_path, markdown_path = write_aggregate(aggregate, tmp_path / "汇总")
    assert aggregate["failure_classification"]["hybrid"]["counts"] == {
        "证据入选但回答失败": 1
    }
    assert aggregate["performance"]["bm25"]["online_retrieval"]["steady_p95_seconds"] == 0.2
    assert aggregate["performance"]["vector"]["chat_calls"]["retry_count"] is None
    assert "未采集" in markdown_path.read_text(encoding="utf-8")
    assert "证据入选但回答失败" in markdown_report(aggregate)
    assert "单题检索（毫秒）" in markdown_report(aggregate)
    assert len([line for line in markdown_report(aggregate).splitlines() if line.startswith("# ")]) == 1
    assert json_path.is_file()
    assert [_sha256(path) for path in paths] == originals
    with pytest.raises(FileExistsError):
        write_aggregate(aggregate, tmp_path / "汇总")


@pytest.mark.parametrize("change,expected", [
    ("question_hash", "Question SHA-256"),
    ("implementation", "implementation"),
    ("embedding", "Embedding"),
    ("chunk", "700/100"),
    ("quality", "quality"),
    ("chat_count", "Chat"),
    ("top5", "Top-5"),
])
def test_aggregate_rejects_mismatched_or_incomplete_runs(
    tmp_path: Path, change: str, expected: str,
) -> None:
    """确认输入或质量不一致时拒绝汇总，且不写任何结果。"""

    questions, paths, reports = _fixture_reports(tmp_path)
    report = reports[1]
    if change == "question_hash":
        report["inputs"]["questions_sha256"] = "wrong"
    elif change == "implementation":
        report["settings"]["implementation_sha256"]["app/evaluation.py"] = "different"
    elif change == "embedding":
        report["settings"]["embedding_identity"]["model_fingerprint"] = "different"
    elif change == "chunk":
        report["spec"]["chunk_size"] = 350
    elif change == "quality":
        report["metrics"]["quality_executed"] = False
    elif change == "chat_count":
        report["performance"]["model_calls"]["chat"]["request_count"] = 0
    elif change == "top5":
        report["cases"][0]["hit_top5"] = False
    paths[1].write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        aggregate_runs(questions, paths)
    assert not (tmp_path / "汇总").exists()


def test_aggregate_rejects_changed_corpus_file(tmp_path: Path) -> None:
    """确认报告彼此一致仍不能掩盖运行后语料已被更改。"""

    questions, paths, _ = _fixture_reports(tmp_path)
    (questions.parent / "规则.txt").write_text("改过的条款", encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus SHA-256 changed"):
        aggregate_runs(questions, paths)
