"""严格核对三轮独立评测，并生成 V0.2 跨策略质量与性能汇总。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any


# 支持直接执行 `python scripts/aggregate_v02_runs.py`。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_v02 import _comparison_markdown, build_comparison, load_cases


STRATEGIES = ("vector", "bm25", "hybrid")
OUTPUT_JSON = "跨策略统一对照.json"
OUTPUT_MARKDOWN = "跨策略统一对照.md"


def _sha256(path: Path) -> str:
    """按原始字节计算输入和逐轮报告的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    """读取逐轮报告并拒绝不完整或错误类型的 JSON。"""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read evaluation report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation report must be a JSON object: {path}.")
    return value


def _require(mapping: dict[str, Any], key: str, label: str) -> Any:
    """指出报告缺失的必填字段，避免空值被当作通过。"""

    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{label} is missing required field {key}.")
    return mapping[key]


def _check_corpus_on_disk(corpus_dir: Path, expected: dict[str, str]) -> None:
    """复算实际语料 SHA，防止只比较报告而忽略运行后文件变动。"""

    root = corpus_dir.resolve()
    if not root.is_dir() or not expected:
        raise ValueError(f"Corpus directory or fingerprints are missing: {corpus_dir}.")
    for relative_name, recorded_sha in expected.items():
        if not isinstance(relative_name, str) or not isinstance(recorded_sha, str):
            raise ValueError("Corpus fingerprints must map relative file names to SHA-256 strings.")
        path = (root / relative_name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Corpus source is missing or outside corpus directory: {relative_name}.")
        if _sha256(path) != recorded_sha:
            raise ValueError(f"Corpus SHA-256 changed after evaluation: {relative_name}.")


def _case_ids(cases: list[dict[str, Any]]) -> set[str]:
    """取唯一题号集合，供逐轮结果完整性校验。"""

    return {case["id"] for case in cases}


def _validate_run(
    report: dict[str, Any], path: Path, questions_sha: str,
    expected_ids: set[str], answerable_count: int, refusal_count: int,
) -> str:
    """校验单轮质量、Top-5 标记、实际 Chat 调用及关键结构。"""

    spec = _require(report, "spec", str(path))
    inputs = _require(report, "inputs", str(path))
    settings = _require(report, "settings", str(path))
    metrics = _require(report, "metrics", str(path))
    outcomes = _require(report, "cases", str(path))
    performance = _require(report, "performance", str(path))
    if not all(isinstance(part, dict) for part in (spec, inputs, settings, metrics, performance)):
        raise ValueError(f"Report has invalid section types: {path}.")
    if not isinstance(outcomes, list) or len(outcomes) != len(expected_ids):
        raise ValueError(f"Report case count differs from questions: {path}.")
    if report.get("schema_version") != 1:
        raise ValueError(f"Unsupported report schema version: {path}.")
    strategy = spec.get("strategy")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy in {path}: {strategy}.")
    if spec.get("chunk_size") != 700 or spec.get("overlap") != 100:
        raise ValueError(f"Report must use frozen 700/100 chunks: {path}.")
    if inputs.get("questions_sha256") != questions_sha:
        raise ValueError(f"Question SHA-256 differs from --questions: {path}.")
    if not isinstance(inputs.get("corpus"), dict) or not inputs["corpus"]:
        raise ValueError(f"Corpus SHA-256 fingerprints are missing: {path}.")
    implementation = settings.get("implementation_sha256")
    if not isinstance(implementation, dict) or not all(
        implementation.get(name) for name in (
            "scripts/evaluate_v02.py", "scripts/evaluation_performance.py",
            "app/document_loader.py", "app/rag.py", "app/evaluation.py",
        )
    ):
        raise ValueError(f"Required implementation SHA-256 fingerprints are missing: {path}.")
    if not isinstance(settings.get("embedding_identity"), dict):
        raise ValueError(f"Embedding identity is missing: {path}.")
    if metrics.get("quality_executed") is not True:
        raise ValueError(f"Answer quality was not executed: {path}.")
    if metrics.get("answerable_count") != answerable_count or metrics.get("refusal_count") != refusal_count:
        raise ValueError(f"Answer/refusal counts differ from questions: {path}.")
    if not all(key in metrics for key in ("top1_hits", "top3_hits", "top5_hits", "acceptance_pass")):
        raise ValueError(f"Top-1/3/5 metrics are missing: {path}.")
    seen: set[str] = set()
    chat_requests = 0
    for outcome in outcomes:
        if not isinstance(outcome, dict) or outcome.get("id") not in expected_ids:
            raise ValueError(f"Report has an unknown or invalid case: {path}.")
        case_id = outcome["id"]
        if case_id in seen:
            raise ValueError(f"Report contains a duplicate case ID: {case_id}.")
        seen.add(case_id)
        if not isinstance(outcome.get("quality_pass"), bool):
            raise ValueError(f"Quality result is missing for {case_id} in {path}.")
        if not isinstance(outcome.get("chat_called"), bool):
            raise ValueError(f"Actual Chat call flag is missing for {case_id} in {path}.")
        count = outcome.get("chat_request_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Actual Chat request count is invalid for {case_id} in {path}.")
        if outcome["chat_called"] != (count > 0):
            raise ValueError(f"Chat call flag disagrees with request count for {case_id}.")
        chat_requests += count
        rank = outcome.get("expected_evidence_rank")
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 1):
            raise ValueError(f"Expected evidence rank is invalid for {case_id}.")
        if outcome.get("hit_top5") != (rank is not None and rank <= 5):
            raise ValueError(f"Top-5 flag disagrees with rank for {case_id}.")
    if seen != expected_ids:
        raise ValueError(f"Report is missing case IDs: {path}.")
    if chat_requests < 1 or metrics.get("chat_request_count") != chat_requests:
        raise ValueError(f"Actual Chat request count is absent or inconsistent: {path}.")
    chat_performance = performance.get("model_calls", {}).get("chat", {})
    if chat_performance.get("request_count") != chat_requests:
        raise ValueError(f"Performance Chat count disagrees with case results: {path}.")
    return strategy


def _shared_identity(report: dict[str, Any]) -> dict[str, Any]:
    """提取除策略名称外必须完全相同的输入和运行配置。"""

    spec = report["spec"]
    return {
        "question_sha256": report["inputs"]["questions_sha256"],
        "corpus_sha256": report["inputs"]["corpus"],
        "corpus_dir": report["inputs"].get("corpus_dir"),
        "settings": report["settings"],
        "spec_without_strategy": {
            key: value for key, value in spec.items() if key not in {"name", "strategy"}
        },
        "relevance_policy": report.get("relevance_policy"),
        "top_k": 5,
    }


def load_validated_runs(
    questions_path: Path, run_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """读取三轮结果，拒绝不同题集、语料、代码、模型或运行配置。"""

    if len(run_paths) != 3 or len({path.resolve() for path in run_paths}) != 3:
        raise ValueError("Exactly three distinct run JSON paths are required.")
    raw_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    mode = "acceptance" if isinstance(raw_questions, list) and len(raw_questions) == 50 else "development"
    cases = load_cases(questions_path, dataset_mode=mode)
    question_sha = _sha256(questions_path)
    expected_ids = _case_ids(cases)
    answerable_count = sum(case["answerable"] for case in cases)
    reports: dict[str, dict[str, Any]] = {}
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    shared: dict[str, Any] | None = None
    for path in run_paths:
        report = _load_json_object(path)
        strategy = _validate_run(
            report, path, question_sha, expected_ids,
            answerable_count, len(cases) - answerable_count,
        )
        if strategy in reports:
            raise ValueError(f"Duplicate strategy report: {strategy}.")
        identity = _shared_identity(report)
        if shared is not None and identity != shared:
            raise ValueError(f"Input, implementation, Embedding or fixed configuration differs: {path}.")
        shared = identity
        reports[strategy] = report
        source_paths[strategy] = path.resolve().as_posix()
        source_hashes[strategy] = _sha256(path)
    if set(reports) != set(STRATEGIES):
        raise ValueError("Runs must contain vector, bm25 and hybrid exactly once.")
    assert shared is not None
    corpus_path = Path(shared["corpus_dir"] or questions_path.parent)
    _check_corpus_on_disk(corpus_path, shared["corpus_sha256"])
    return cases, [reports[strategy] for strategy in STRATEGIES], {
        "questions_path": questions_path.resolve().as_posix(),
        "run_paths": source_paths,
        "run_sha256": source_hashes,
        **shared,
    }


def _performance_summary(report: dict[str, Any]) -> dict[str, Any]:
    """抽取可比较的分阶段时间、单题百分位、内存和真实模型请求。"""

    performance = report["performance"]
    stages = performance.get("stages", {})
    latencies = performance.get("latency", {})
    memory = performance.get("memory", {})
    calls = performance.get("model_calls", {})
    return {
        "total_seconds": performance.get("total_seconds"),
        "hardware": performance.get("hardware"),
        "initialization": stages.get("initialization", {"status": "未采集"}),
        "model_load": stages.get("model_load", {"status": "未采集"}),
        "parse_chunk": stages.get("parse_chunk", {"status": "未采集"}),
        "vectorize": stages.get("vectorize", {"status": "未采集"}),
        "index_build": stages.get("index_build", {"status": "未采集"}),
        "chat_client_setup": stages.get("chat_client_setup", {"status": "未采集"}),
        "online_retrieval": latencies.get("online_retrieval", {"status": "未采集"}),
        "chat_latency": latencies.get("chat", {"status": "未采集"}),
        "online_total": latencies.get("online_total", {"status": "未采集"}),
        "process_peak_rss_bytes": memory.get("process_peak_rss_bytes", {"status": "未采集"}),
        "gpu_process_peak_allocated_bytes": memory.get(
            "gpu_process_peak_allocated_bytes", {"status": "未采集"}
        ),
        "embedding_calls": calls.get("embedding", {"status": "未采集"}),
        "chat_calls": calls.get("chat", {"status": "未采集"}),
        "measurement_note": performance.get("metadata", {}).get(
            "online_total_note", "在线端到端问答耗时未采集"
        ),
    }


def _classify_failure(case: dict[str, Any], outcome: dict[str, Any]) -> str | None:
    """按执行、复核、召回和回答顺序为每题指定主要失败原因。"""

    if outcome.get("error") or outcome.get("quality_error") or outcome.get("manual_review_status") == "执行错误":
        return "执行错误"
    if outcome.get("manual_review_status") == "待人工复核":
        return "待人工复核"
    if case["answerable"]:
        if not outcome.get("hit_top5"):
            return "证据在前5之后" if outcome.get("expected_evidence_rank") is not None else "证据未召回"
        if outcome.get("quality_pass") is not True:
            return "证据入选但回答失败"
    elif outcome.get("quality_pass") is not True:
        return "拒答质量失败"
    return None


def _failure_summary(
    cases: list[dict[str, Any]], report: dict[str, Any],
) -> dict[str, Any]:
    """保留失败题号、主要原因和候选片段，供后续人工复核。"""

    by_id = {outcome["id"]: outcome for outcome in report["cases"]}
    details: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for case in cases:
        outcome = by_id[case["id"]]
        category = _classify_failure(case, outcome)
        if category is None:
            continue
        counts[category] = counts.get(category, 0) + 1
        details.append({
            "id": case["id"],
            "category": category,
            "answerable": case["answerable"],
            "expected_source": case["expected_source"],
            "expected_evidence_rank": outcome.get("expected_evidence_rank"),
            "hit_top5": outcome.get("hit_top5"),
            "quality_pass": outcome.get("quality_pass"),
            "manual_review_status": outcome.get("manual_review_status"),
            "error": outcome.get("quality_error") or outcome.get("error"),
            "unsupported_numeric_claims": outcome.get("unsupported_numeric_claims", []),
            "unverified_claims": outcome.get("unverified_claims", []),
            "top5_candidates": [
                {
                    "rank": candidate.get("rank"),
                    "file_name": candidate.get("file_name"),
                    "page_number": candidate.get("page_number"),
                    "chunk_id": candidate.get("chunk_id"),
                    "text": candidate.get("text"),
                }
                for candidate in outcome.get("ranking", [])
                if isinstance(candidate, dict) and isinstance(candidate.get("rank"), int)
                and candidate["rank"] <= 5
            ],
        })
    return {"counts": counts, "cases": details}


def aggregate_runs(
    questions_path: Path, run_paths: list[Path],
) -> dict[str, Any]:
    """先完成一致性校验，再复用原对照算法生成统一质量与性能结果。"""

    cases, reports, inputs = load_validated_runs(questions_path, run_paths)
    comparison = build_comparison(reports, cases)
    comparison["generated_on"] = date.today().isoformat()
    performance = {
        report["spec"]["strategy"]: _performance_summary(report)
        for report in reports
    }
    failures = {
        report["spec"]["strategy"]: _failure_summary(cases, report)
        for report in reports
    }
    return {
        "schema_version": 1,
        "generated_on": date.today().isoformat(),
        "validation": {
            "strategies": list(STRATEGIES),
            "chunk_size": 700,
            "overlap": 100,
            "quality_reference_top_k": 5,
            "all_quality_executed": True,
            "all_acceptance_pass": all(
                report["metrics"]["acceptance_pass"] is True for report in reports
            ),
        },
        "inputs": inputs,
        "comparison": comparison,
        "performance": performance,
        "failure_classification": failures,
    }


def _display_metric(value: Any, digits: int = 3) -> str:
    """将缺失数值明确显示为未采集，避免和真实零值混淆。"""

    if value is None:
        return "未采集"
    if isinstance(value, dict):
        if value.get("status") != "已采集":
            return "未采集"
        value = value.get("value")
        if value is None:
            return "未采集"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _stage_seconds(summary: dict[str, Any], name: str) -> str:
    """格式化一个阶段总耗时并保留未采集状态。"""

    stage = summary[name]
    return _display_metric(stage.get("total_seconds") if stage.get("status") == "已采集" else None)


def _latency_cell(summary: dict[str, Any], name: str, *, milliseconds: bool = False) -> str:
    """按指定单位显示首题与常态延迟，保留快速检索的非零数值。"""

    latency = summary[name]
    if latency.get("status") != "已采集":
        return "未采集"
    scale = 1000 if milliseconds else 1

    def display(value: float | None) -> str:
        """把秒换算成显示单位，并沿用缺失值标记。"""

        return _display_metric(value * scale if value is not None else None)

    return (
        f"首题 {display(latency.get('first_seconds'))}；"
        f"常态 {display(latency.get('steady_p50_seconds'))}/"
        f"{display(latency.get('steady_p95_seconds'))}；"
        f"n={latency.get('steady_sample_count', 0)}"
    )


def _model_cell(summary: dict[str, Any], name: str) -> str:
    """展示真实模型请求、失败、重试及服务端 token 用量。"""

    calls = summary[name]
    if calls.get("status") != "已采集":
        return "未采集"
    return (
        f"请求 {_display_metric(calls.get('request_count'))}；"
        f"失败 {_display_metric(calls.get('failure_count'))}；"
        f"重试 {_display_metric(calls.get('retry_count'))}；"
        f"token {_display_metric(calls.get('total_tokens'))}"
    )


def markdown_report(aggregate: dict[str, Any]) -> str:
    """在原检索质量对照后追加性能表和可定位的失败分类。"""

    comparison = aggregate["comparison"]
    # 复用既有对照正文时降低标题层级，使整份报告只有一个一级标题。
    comparison_lines = _comparison_markdown(comparison).splitlines()
    comparison_lines[0] = "## 检索与回答质量"
    comparison_section = "\n".join(
        f"#{line}" if line.startswith("## ") and index > 0 else line
        for index, line in enumerate(comparison_lines)
    )
    lines = [
        "# TraceRAG V0.2 业务泛化跨策略统一对照",
        "",
        f"- 日期：{aggregate['generated_on']}",
        f"- 题集 SHA-256：`{aggregate['inputs']['question_sha256']}`",
        "- 输入、代码、Embedding 身份、700/100 切分与 Top-5 均已通过一致性检查；三策略均实际执行回答质量。",
        f"- 整体验收：{'三策略均通过' if aggregate['validation']['all_acceptance_pass'] else '至少一个策略未通过'}。",
        "- 在线单题检索为独立 Top-5 探针；质量评估使用预计算排名。在线端到端问答耗时未采集。",
        "",
        comparison_section,
        "",
        "## 性能与资源",
        "",
        "总耗时、初始化、模型加载和 Chat 延迟单位为秒；单题检索延迟单位为毫秒，峰值 RSS 单位为 MiB。首题与常态样本分列。",
        "",
        "| 策略 | 总耗时（秒） | 初始化（秒） | 本地模型加载（秒） | 单题检索（毫秒）：首题 / 常态 P50/P95 / n | Chat（秒）：首题 / 常态 P50/P95 / n | 峰值 RSS（MiB） |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for strategy in STRATEGIES:
        item = aggregate["performance"][strategy]
        rss = item["process_peak_rss_bytes"]
        rss_mib = (
            rss.get("value") / (1024 * 1024)
            if rss.get("status") == "已采集" and isinstance(rss.get("value"), (int, float))
            else None
        )
        lines.append(
            f"| {strategy} | {_display_metric(item['total_seconds'])} | "
            f"{_stage_seconds(item, 'initialization')} | {_stage_seconds(item, 'model_load')} | "
            f"{_latency_cell(item, 'online_retrieval', milliseconds=True)} | {_latency_cell(item, 'chat_latency')} | "
            f"{_display_metric(rss_mib, 1)} |"
        )
    lines.extend([
        "",
        "| 策略 | Embedding 调用 | Chat 调用 | 在线端到端 |",
        "| --- | --- | --- | --- |",
    ])
    for strategy in STRATEGIES:
        item = aggregate["performance"][strategy]
        lines.append(
            f"| {strategy} | {_model_cell(item, 'embedding_calls')} | "
            f"{_model_cell(item, 'chat_calls')} | "
            f"{'未采集' if item['online_total'].get('status') != '已采集' else _latency_cell(item, 'online_total')} |"
        )
    lines.extend(["", "## 逐题失败分类", ""])
    for strategy in STRATEGIES:
        failure = aggregate["failure_classification"][strategy]
        lines.append(f"### {strategy}")
        lines.append("")
        if not failure["cases"]:
            lines.extend(["- 无失败题。", ""])
            continue
        for category, count in failure["counts"].items():
            ids = [item["id"] for item in failure["cases"] if item["category"] == category]
            lines.append(f"- {category}（{count}）：{', '.join(ids)}")
        lines.append("")
    lines.extend([
        "逐题候选片段、执行错误和待复核状态保存在本目录 JSON；原始逐轮报告路径及 SHA-256 亦保存在 JSON。",
        "",
    ])
    return "\n".join(lines)


def write_aggregate(aggregate: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """只写新的统一结果文件，拒绝覆盖已有汇总或原始报告。"""

    json_path = output_dir / OUTPUT_JSON
    markdown_path = output_dir / OUTPUT_MARKDOWN
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(f"Cross-run report already exists in {output_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown_report(aggregate), encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    """解析三份逐轮 JSON，严格核对后输出统一对照。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs=3, required=True,
                        metavar=("VECTOR_JSON", "BM25_JSON", "HYBRID_JSON"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    aggregate = aggregate_runs(args.questions, args.runs)
    json_path, markdown_path = write_aggregate(aggregate, args.output_dir)
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
