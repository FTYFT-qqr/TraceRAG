"""为评测记录阶段耗时、逐题延迟、资源占用和实际模型调用。"""

from __future__ import annotations

import ctypes
import functools
import math
import os
import platform
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar


_CREATE_RESULT = TypeVar("_CREATE_RESULT")
_STAGES = (
    "model_load", "parse_chunk", "vectorize", "index_build",
    "initialization", "offline_retrieval", "chat",
)
_LATENCIES = ("offline_retrieval", "online_retrieval", "chat", "online_total")
_MODEL_KINDS = ("embedding", "chat")


def _metric(value: int | float | None, reason: str = "数据源不可用") -> dict[str, Any]:
    """给可选数值附带采集状态，防止把缺失值误写为零。"""

    return {"status": "已采集", "value": value} if value is not None else {
        "status": "未采集", "value": None, "reason": reason,
    }


def _percentile(values: list[float], percentage: float) -> float | None:
    """用线性插值计算百分位，空样本返回缺失值。"""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _process_memory_bytes() -> tuple[int | None, int | None]:
    """优先读取系统报告的进程 RSS 和峰值，不引入额外依赖。"""

    try:
        if sys.platform == "win32":
            size_t = ctypes.c_size_t

            class ProcessMemoryCounters(ctypes.Structure):
                """映射 Windows PROCESS_MEMORY_COUNTERS 中的工作集字段。"""

                _fields_ = [
                    ("cb", ctypes.c_ulong), ("page_fault_count", ctypes.c_ulong),
                    ("peak_working_set_size", size_t), ("working_set_size", size_t),
                    ("quota_peak_paged_pool_usage", size_t),
                    ("quota_paged_pool_usage", size_t),
                    ("quota_peak_non_paged_pool_usage", size_t),
                    ("quota_non_paged_pool_usage", size_t),
                    ("pagefile_usage", size_t), ("peak_pagefile_usage", size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process.restype = ctypes.c_void_p
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
            handle = get_process()
            if get_memory(
                handle, ctypes.byref(counters), counters.cb
            ):
                return int(counters.working_set_size), int(counters.peak_working_set_size)
        elif sys.platform.startswith("linux"):
            with open("/proc/self/statm", encoding="ascii") as statm_file:
                fields = statm_file.read().split()
            rss = int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
            peak = None
            with open("/proc/self/status", encoding="ascii") as status_file:
                for line in status_file:
                    if line.startswith("VmHWM:"):
                        peak = int(line.split()[1]) * 1024
                        break
            return rss, peak
    except (OSError, ValueError, AttributeError, IndexError):
        pass
    return None, None


def _gpu_memory_bytes() -> dict[str, int] | None:
    """仅在已加载的 PyTorch 能访问 CUDA 时读取实际显存。"""

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {
            "free_bytes": int(free),
            "total_bytes": int(total),
            "process_allocated_bytes": int(torch.cuda.memory_allocated()),
            "process_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
    except (AttributeError, RuntimeError, OSError):
        return None


def _usage_field(response: Any, name: str) -> int | None:
    """从兼容 API 的响应 usage 中读取服务端实际返回的 token 数。"""

    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class PerformanceRecorder:
    """独立保存性能观测；业务代码显式指定采样边界和工作负载。"""

    def __init__(
        self,
        metadata: dict[str, Any] | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """开始一次运行，并读取不含密钥的本机硬件标识。"""

        self.metadata = dict(metadata or {})
        self._clock = clock or time.perf_counter
        self._started = self._clock()
        self._ended: float | None = None
        self._stages: dict[str, list[float]] = defaultdict(list)
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._observed_models: set[str] = set()
        self._rss_samples: list[int] = []
        self._process_peak_samples: list[int] = []
        self._gpu_sample: dict[str, int] | None = None
        self._hardware = {
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
        }
        self._sample_memory()

    def _sample_memory(self) -> None:
        """尽力保存资源快照；采集失败不会改变评测业务结果。"""

        try:
            rss, peak = _process_memory_bytes()
            if rss is not None:
                self._rss_samples.append(rss)
            if peak is not None:
                self._process_peak_samples.append(peak)
        except Exception:
            pass
        try:
            gpu = _gpu_memory_bytes()
            if gpu is not None:
                self._gpu_sample = gpu
        except Exception:
            pass

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """记录一个阶段的墙钟耗时，异常时也保存已用时间。"""

        if not name.strip():
            raise ValueError("Stage name cannot be empty.")
        started = self._clock()
        try:
            yield
        finally:
            self._stages[name].append(max(0.0, self._clock() - started))
            self._sample_memory()

    def record_query_latency(self, *, kind: str, duration_seconds: float) -> None:
        """显式记录离线或在线单题耗时；第一题自动与常态分开。"""

        if kind not in _LATENCIES:
            raise ValueError(f"Unsupported latency kind: {kind}.")
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError("Latency must be a finite non-negative number.")
        self._latencies[kind].append(duration_seconds)

    def mark_model_observed(self, kind: str) -> None:
        """声明真实模型调用入口已接入，以便零调用可如实记录为零。"""

        if kind not in _MODEL_KINDS:
            raise ValueError(f"Unsupported model kind: {kind}.")
        self._observed_models.add(kind)

    def record_model_call(
        self,
        *,
        kind: str,
        success: bool,
        retries: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """只记录实际调用边界；未知重试及 token 用量保留为空。"""

        self.mark_model_observed(kind)
        self._calls[kind].append({
            "success": bool(success),
            "retries": retries,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_seconds": duration_seconds,
        })

    def finish(self) -> None:
        """固定总耗时；多次调用不会延长已经结束的运行。"""

        if self._ended is None:
            self._ended = self._clock()
            self._sample_memory()

    def report(self) -> dict[str, Any]:
        """生成可序列化报告，缺失字段明确标记未采集。"""

        self.finish()
        stage_names = dict.fromkeys((*_STAGES, *self._stages))
        stages = {}
        for name in stage_names:
            samples = self._stages.get(name, [])
            stages[name] = {
                "status": "已采集" if samples else "未采集",
                "sample_count": len(samples),
                "total_seconds": sum(samples) if samples else None,
            }
        latency = {}
        for kind in _LATENCIES:
            samples = self._latencies.get(kind, [])
            steady = samples[1:]
            latency[kind] = {
                "status": "已采集" if samples else "未采集",
                "sample_count": len(samples),
                "first_seconds": samples[0] if samples else None,
                "steady_sample_count": len(steady),
                "steady_p50_seconds": _percentile(steady, 50),
                "steady_p95_seconds": _percentile(steady, 95),
            }
        calls = {}
        for kind in _MODEL_KINDS:
            events = self._calls.get(kind, [])
            observed = kind in self._observed_models
            call_metrics = {
                "status": "已采集" if observed else "未采集",
                "request_count": len(events) if observed else None,
                "failure_count": sum(not event["success"] for event in events) if observed else None,
                "retry_count": _sum_complete(events, "retries"),
                "prompt_tokens": _sum_complete(events, "prompt_tokens"),
                "completion_tokens": _sum_complete(events, "completion_tokens"),
                "total_tokens": _sum_complete(events, "total_tokens"),
                "request_duration_seconds": _sum_complete(events, "duration_seconds"),
                "token_usage_observed_count": sum(
                    event["total_tokens"] is not None for event in events
                ) if observed else None,
                "retry_observed_count": sum(
                    event["retries"] is not None for event in events
                ) if observed else None,
            }
            for field in (
                "retry_count", "prompt_tokens", "completion_tokens",
                "total_tokens", "request_duration_seconds",
            ):
                # 已接入 SDK 但服务未返回 usage，也属于明确未采集。
                call_metrics[f"{field}_status"] = (
                    "已采集" if call_metrics[field] is not None else "未采集"
                )
            calls[kind] = call_metrics
        gpu = self._gpu_sample or {}
        return {
            "schema_version": 1,
            "metadata": self.metadata,
            "hardware": self._hardware,
            "total_seconds": max(0.0, self._ended - self._started),
            "stages": stages,
            "latency": latency,
            "memory": {
                "process_rss_bytes": _metric(self._rss_samples[-1] if self._rss_samples else None),
                "process_peak_rss_bytes": _metric(
                    max((*self._rss_samples, *self._process_peak_samples), default=None)
                ),
                "gpu_free_bytes": _metric(gpu.get("free_bytes"), "CUDA 显存不可用"),
                "gpu_total_bytes": _metric(gpu.get("total_bytes"), "CUDA 显存不可用"),
                "gpu_process_allocated_bytes": _metric(
                    gpu.get("process_allocated_bytes"), "CUDA 显存不可用"
                ),
                "gpu_process_peak_allocated_bytes": _metric(
                    gpu.get("process_peak_allocated_bytes"), "CUDA 显存不可用"
                ),
            },
            "model_calls": calls,
        }


def _sum_complete(events: list[dict[str, Any]], key: str) -> int | float | None:
    """仅在所有请求均报告某指标时汇总，避免部分数据冒充总数。"""

    values = [event[key] for event in events]
    return sum(values) if values and all(value is not None for value in values) else None


def observe_model_create(
    create: Callable[..., _CREATE_RESULT],
    recorder: PerformanceRecorder,
    *,
    kind: str,
) -> Callable[..., _CREATE_RESULT]:
    """包裹实际 SDK create 调用，透明记录成功、失败和服务端 usage。"""

    recorder.mark_model_observed(kind)

    @functools.wraps(create)
    def observed(*args: Any, **kwargs: Any) -> _CREATE_RESULT:
        """原样转发参数与结果；内部重试不可见时不推测次数。"""

        started = time.perf_counter()
        try:
            response = create(*args, **kwargs)
        except Exception:
            try:
                recorder.record_model_call(
                    kind=kind, success=False,
                    duration_seconds=time.perf_counter() - started,
                )
            except Exception:
                pass
            raise
        try:
            recorder.record_model_call(
                kind=kind, success=True,
                prompt_tokens=_usage_field(response, "prompt_tokens"),
                completion_tokens=_usage_field(response, "completion_tokens"),
                total_tokens=_usage_field(response, "total_tokens"),
                duration_seconds=time.perf_counter() - started,
            )
        except Exception:
            pass
        return response

    return observed
