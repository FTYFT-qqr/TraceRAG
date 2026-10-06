"""验证性能证据不会把缺失值、离线耗时和拒答误记为模型调用。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import evaluation_performance
from scripts.evaluation_performance import PerformanceRecorder, observe_model_create


def test_first_query_and_offline_latency_remain_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    """首题、常态分位数与离线批量检索必须各自独立。"""

    monkeypatch.setattr(evaluation_performance, "_process_memory_bytes", lambda: (None, None))
    monkeypatch.setattr(evaluation_performance, "_gpu_memory_bytes", lambda: None)
    ticks = iter((0.0, 1.0, 3.0, 4.0))
    recorder = PerformanceRecorder(clock=lambda: next(ticks))
    with recorder.stage("parse_chunk"):
        pass
    for seconds in (3.0, 1.0, 2.0):
        recorder.record_query_latency(kind="offline_retrieval", duration_seconds=seconds)
    result = recorder.report()

    assert result["total_seconds"] == 4.0
    assert result["stages"]["parse_chunk"]["total_seconds"] == 2.0
    offline = result["latency"]["offline_retrieval"]
    assert offline["first_seconds"] == 3.0
    assert offline["steady_sample_count"] == 2
    assert offline["steady_p50_seconds"] == 1.5
    assert offline["steady_p95_seconds"] == pytest.approx(1.95)
    assert result["latency"]["online_retrieval"]["status"] == "未采集"
    assert result["latency"]["online_retrieval"]["first_seconds"] is None
    assert result["memory"]["process_rss_bytes"]["value"] is None
    assert result["memory"]["gpu_total_bytes"]["status"] == "未采集"
    assert result["model_calls"]["chat"]["request_count"] is None


def test_actual_create_calls_and_partial_usage_are_reported() -> None:
    """只有真正调用 SDK create 才计请求，失败照记且未知重试保持空值。"""

    recorder = PerformanceRecorder()
    request_count = 0

    def create(*, fail: bool = False) -> SimpleNamespace:
        """模拟成功和抛错的模型 API 请求。"""

        nonlocal request_count
        request_count += 1
        if fail:
            raise RuntimeError("remote failure")
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10))

    observed = observe_model_create(create, recorder, kind="chat")
    assert recorder.report()["model_calls"]["chat"]["request_count"] == 0
    assert observed().usage.total_tokens == 10
    with pytest.raises(RuntimeError, match="remote failure"):
        observed(fail=True)
    calls = recorder.report()["model_calls"]["chat"]
    assert request_count == 2
    assert calls["request_count"] == 2
    assert calls["failure_count"] == 1
    assert calls["retry_count"] is None
    assert calls["retry_count_status"] == "未采集"
    assert calls["retry_observed_count"] == 0
    assert calls["total_tokens"] is None
    assert calls["total_tokens_status"] == "未采集"
    assert calls["token_usage_observed_count"] == 1


def test_stage_exception_keeps_timing_and_original_error() -> None:
    """阶段中的业务异常原样传播，同时留存已经发生的耗时。"""

    ticks = iter((0.0, 0.5, 1.5, 2.0))
    recorder = PerformanceRecorder(clock=lambda: next(ticks))
    with pytest.raises(ValueError, match="source failed"):
        with recorder.stage("parse_chunk"):
            raise ValueError("source failed")
    result = recorder.report()
    assert result["stages"]["parse_chunk"]["total_seconds"] == 1.0
    assert result["model_calls"]["chat"]["status"] == "未采集"


def test_memory_probe_failure_does_not_interrupt_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    """系统资源接口报错时，评测仍继续并明确标记内存未采集。"""

    def unavailable() -> None:
        """模拟系统拒绝读取资源计数器。"""

        raise RuntimeError("counter unavailable")

    monkeypatch.setattr(evaluation_performance, "_process_memory_bytes", unavailable)
    monkeypatch.setattr(evaluation_performance, "_gpu_memory_bytes", unavailable)
    recorder = PerformanceRecorder()
    with recorder.stage("parse_chunk"):
        pass
    report = recorder.report()
    assert report["stages"]["parse_chunk"]["status"] == "已采集"
    assert report["memory"]["process_rss_bytes"]["status"] == "未采集"
    assert report["memory"]["gpu_total_bytes"]["value"] is None
