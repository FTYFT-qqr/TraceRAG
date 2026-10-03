"""验证 V0.2 评测集 schema 和报告核心数据结构。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from argparse import Namespace

from app.config import Settings
from scripts.evaluate_v02 import (
    BenchmarkSpec,
    _comparison_markdown,
    _embedding_comparison_settings,
    _retrieval_outcome,
    build_comparison,
    load_cases,
    validate_cases_against_corpus,
)
from app.models import Chunk, RetrievalResult
from app.config import Settings
from scripts import evaluate_v02


QUESTION_FILE = Path("samples/acceptance/questions_v02.json")


def test_v02_questions_have_fifty_cases_and_required_categories() -> None:
    """确认评测集数量和规划要求的多类场景覆盖。"""

    cases = load_cases(QUESTION_FILE)

    assert len(cases) == 50
    assert sum(case["answerable"] for case in cases) == 43
    assert sum("supported_negative" in case["tags"] for case in cases) >= 8
    assert sum("numeric" in case["tags"] for case in cases) >= 10
    assert sum("multi_fact" in case["tags"] for case in cases) >= 10
    assert sum(not case["answerable"] for case in cases) == 7
    validate_cases_against_corpus(cases)


def test_v02_schema_rejects_duplicate_ids(tmp_path: Path) -> None:
    """确认重复题号会阻止固定评测集加载。"""

    cases = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))
    cases[1]["id"] = cases[0]["id"]
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_cases(path)


def test_retrieval_outcome_keeps_full_ranking_and_top_k_flags() -> None:
    """确认排名输出保留全部候选并正确计算 Top-1/3/5。"""

    case = {
        "answerable": True,
        "expected_source": "policy.txt",
        "expected_evidence": "目标证据",
    }
    chunks = [
        Chunk(f"chunk-{index}", "doc", "目标证据" if index == 2 else "其他", "policy.txt", None, index)
        for index in range(5)
    ]
    results = [
        RetrievalResult(chunk, 1.0 - index / 10, candidate_rank=index + 1)
        for index, chunk in enumerate(chunks)
    ]

    outcome = _retrieval_outcome(case, results)

    assert outcome["expected_evidence_rank"] == 3
    assert outcome["hit_top1"] is False
    assert outcome["hit_top3"] is True
    assert outcome["hit_top5"] is True
    assert len(outcome["ranking"]) == 5
    assert outcome["ranking"][2]["chunk_id"] == "chunk-2"


def test_benchmark_spec_keeps_single_variable_fields() -> None:
    """确认切分对照只改变指定参数，召回策略保持不变。"""

    baseline = BenchmarkSpec("baseline", "vector", 700, 100)
    short = BenchmarkSpec("short", "vector", 350, 50)

    assert baseline.strategy == short.strategy
    assert baseline.chunk_size != short.chunk_size
    assert baseline.overlap != short.overlap


def test_embedding_batch_comparison_changes_only_batch_size(tmp_path: Path) -> None:
    """确认 Embedding 批次对照固定模型身份，只变更单个推理参数。"""

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "weights.bin").write_bytes(b"model")
    settings = Settings(
        api_key=None,
        embedding_provider="local",
        local_embedding_path=str(model_path),
    )
    args = Namespace(
        compare_embedding_model=None,
        compare_embedding_path=None,
        compare_embedding_batch_size=4,
        embedding_batch_size=1,
    )

    runs = _embedding_comparison_settings(settings, args)

    assert [size for _, _, size in runs] == [1, 4]
    assert runs[0][1].embedding_identity == runs[1][1].embedding_identity


def test_single_quality_run_summary_keeps_bad_cases_and_quality_counts() -> None:
    """确认单轮质量报告也列出基线漏检题及事实、引用、拒答指标。"""

    result = {
        "spec": {"name": "quality", "strategy": "vector", "chunk_size": 700, "overlap": 100, "embedding_batch_size": 1},
        "cases": [{"id": "miss", "hit_top5": False}, {"id": "refusal", "hit_top5": False}],
        "metrics": {
            "top1_hits": 0,
            "top3_hits": 0,
            "top5_hits": 0,
            "answerable_count": 1,
            "quality_pass": 0,
            "citation_traceable": 1,
            "citation_original_support": 0,
            "refusal_pass": 1,
            "refusal_count": 1,
            "manual_review_count": 0,
            "quality_executed": True,
            "acceptance_pass": False,
        },
    }

    summary = build_comparison([result], [{"id": "miss", "answerable": True}, {"id": "refusal", "answerable": False}])
    markdown = _comparison_markdown(summary)

    assert summary["runs"][0]["top5_bad_case_ids"] == ["miss"]
    assert "miss" in markdown
    assert "## 回答质量" in markdown
    assert "1/1" in markdown


@pytest.mark.parametrize("quality,overrides,expected", [
    (True, {}, 0),
    (True, {"acceptance_pass": False, "quality_pass": 0}, 1),
    (True, {"acceptance_pass": False, "manual_review_count": 1}, 1),
    (True, {"acceptance_pass": False, "refusal_pass": 0}, 1),
    (False, {"quality_executed": False, "acceptance_pass": False}, 0),
    (False, {"quality_executed": False, "acceptance_pass": False, "retrieval_pass": False}, 1),
    (False, {"quality_executed": False, "acceptance_pass": False, "errors": 1}, 1),
    (True, {"quality_executed": False, "acceptance_pass": False}, 1),
])
def test_cli_exit_reflects_actual_acceptance(
    monkeypatch, tmp_path: Path, quality: bool, overrides: dict, expected: int
) -> None:
    """通过 CLI 主流程验证质量失败、复核、拒答、检索门槛及错误均返回失败码。"""

    metrics = {
        "top1_hits": 10, "top3_hits": 10, "top5_hits": 10,
        "answerable_count": 1, "quality_pass": 1,
        "citation_traceable": 1, "citation_original_support": 1,
        "refusal_pass": 1, "refusal_count": 1, "manual_review_count": 0,
        "errors": 0, "retrieval_pass": True, "quality_executed": True,
        "acceptance_pass": True, **overrides,
    }
    result = {
        "spec": {"name": "cli", "strategy": "bm25", "chunk_size": 700, "overlap": 100, "embedding_batch_size": 1},
        "cases": [{"id": "cli", "hit_top5": True}], "metrics": metrics,
    }
    args = SimpleNamespace(
        chunk_size=700, overlap=100, questions=tmp_path / "questions.json", suite=False,
        name="cli", strategy="bm25", rrf_k=60, quality=quality, output_dir=tmp_path,
    )
    monkeypatch.setattr(evaluate_v02, "_parse_args", lambda: args)
    monkeypatch.setattr(evaluate_v02, "_settings_from_args", lambda args: Settings())
    monkeypatch.setattr(evaluate_v02, "load_cases", lambda path: [{"id": "cli", "answerable": True}])
    monkeypatch.setattr(evaluate_v02, "validate_cases_against_corpus", lambda cases: None)
    monkeypatch.setattr(evaluate_v02, "_embedding_comparison_settings", lambda settings, args: [("baseline", settings, 1)])
    monkeypatch.setattr(evaluate_v02, "run_benchmark", lambda *args, **kwargs: result)
    monkeypatch.setattr(evaluate_v02, "_write_result", lambda *args: (tmp_path / "运行.json", tmp_path / "运行.md"))
    assert evaluate_v02.main() == expected
    assert (tmp_path / "对照摘要.json").is_file()
