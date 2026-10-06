"""验收 Compose 中的真实模型、公开资料问答及跨容器重启恢复。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import requests

from app import __version__
from app.evaluation import assess_case


ROOT = Path(__file__).resolve().parents[1]
QUESTION_IDS = (
    "LIB-V2-01", "LIB-V2-10", "LAB-V2-07", "CLINIC-V2-13", "LIB-V2-12",
)
SOURCES = ("library_service.txt", "lab_equipment.txt", "clinic_service.txt")


def request_json(method: str, url: str, **kwargs) -> dict:
    """通过实际 HTTP 接口请求结果，不让宿主代理接管本地验收流量。"""
    with requests.Session() as session:
        session.trust_env = False
        response = session.request(method, url, timeout=300, **kwargs)
        response.raise_for_status()
        return response.json()


def wait_healthy(api_url: str, ui_url: str, timeout: float = 180) -> dict:
    """等待 API 和 UI 恢复健康，重启后的暂时断连允许重试。"""
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            health = request_json("GET", f"{api_url}/health")
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(f"{ui_url}/_stcore/health", timeout=5)
                response.raise_for_status()
            if health["status"] == "ok" and health["version"] == __version__ and response.text.strip() == "ok":
                return health
        except (requests.RequestException, KeyError) as exc:
            last_error = type(exc).__name__
        time.sleep(1)
    raise RuntimeError(f"容器健康检查超时：{last_error}")


def assess_answer(question: dict, response: dict) -> dict:
    """只适配 HTTP 数据形状，复用正式50题评测的公开质量判断入口。"""
    # HTTP 没有暴露文档内部编号；评测所需的来源、正文和名次直接取实际响应。
    results = [
        SimpleNamespace(
            score=hit["score"], candidate_rank=hit["candidate_rank"],
            chunk=SimpleNamespace(
                chunk_id=hit["chunk_id"], file_name=hit["file_name"],
                page_number=hit["page_number"], content=hit["text"],
            ),
        )
        for hit in response["retrievals"]
    ]
    converted = SimpleNamespace(
        answer=response["answer"], rejected=response["rejected"],
        citations=[SimpleNamespace(**citation) for citation in response["citations"]],
        results=results,
    )
    return assess_case(question, converted)


def answer_passes(question: dict, response: dict) -> bool:
    """拒答、事实、逐句引用和待人工复核须满足正式质量验收条件。"""
    return bool(assess_answer(question, response)["quality_pass"])


def run(api_url: str, ui_url: str, output: Path, project: str, docker: str) -> dict:
    """在隔离 Compose 项目中导入三份样例，保存重启前后的真实响应。"""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", project):
        raise ValueError("Compose 项目名只能包含小写字母、数字、下划线和连字符。")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "mode": "compose-real-provider",
        "scope": "三份公开样例、五个定向质量场景和跨容器重启，不替代完整50题评测。",
        "compose_project": project,
        "checks": [],
        "passed": False,
    }

    def record(name: str, passed: bool, detail: object) -> None:
        """即时保存每项证据，使中断后仍可追溯已执行的检查。"""
        report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        save()
        if not passed:
            raise AssertionError(name)

    def save() -> None:
        """只写验收证据，不记录 API 密钥或容器环境变量。"""
        (output / "真实模型与重启验收.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    def upload(name: str) -> dict:
        """读取版本控制内的公开资料，并上传至容器的独立索引卷。"""
        content = (ROOT / "samples" / "acceptance" / name).read_bytes()
        return request_json("POST", f"{api_url}/documents", files={"file": (name, content, "text/plain")})

    def query_cases(stage: str, questions: list[dict]) -> None:
        """分别检查普通事实、否定事实和缺少金额证据时的拒答。"""
        for question in questions:
            response = request_json("POST", f"{api_url}/query", json={"query": question["query"], "top_k": 5})
            quality = assess_answer(question, response)
            record(f"{stage}:{question['id']}", quality["quality_pass"], {"question": question, "response": response, "quality": quality})

    try:
        record("API和UI健康", True, wait_healthy(api_url, ui_url))
        generation = ""
        for name in SOURCES:
            result = upload(name)
            generation = result["index_generation"]
            record(f"上传:{name}", result["chunk_count"] > 0, result)
        questions = json.loads((ROOT / "samples" / "acceptance" / "questions_v02.json").read_text(encoding="utf-8"))
        cases = [next(row for row in questions if row["id"] == question_id) for question_id in QUESTION_IDS]
        query_cases("重启前", cases)
        # 只重启指定项目的 API，保留索引卷，避免干扰其他项目和本机服务。
        command = [docker, "compose", "--project-name", project, "-f", str(ROOT / "compose.yaml"), "-f", str(ROOT / "compose.local.yaml"), "restart", "api"]
        restarted = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        record("重启API容器", restarted.returncode == 0, {"exit_code": restarted.returncode})
        record("重启后API和UI健康", True, wait_healthy(api_url, ui_url))
        # 先查询再上传，证明恢复来自已保存的卷，而非再次导入资料。
        query_cases("重启后", cases)
        for name in SOURCES:
            result = upload(name)
            record(f"重启后重复跳过:{name}", result["already_indexed"] and result["index_generation"] == generation, result)
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        save()
    return report


def main() -> None:
    """解析容器验收参数，任何质量或恢复检查失败都返回非零退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:18000")
    parser.add_argument("--ui-url", default="http://127.0.0.1:18501")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compose-project", required=True)
    parser.add_argument("--docker-executable", default="docker")
    arguments = parser.parse_args()
    report = run(arguments.api_url.rstrip("/"), arguments.ui_url.rstrip("/"), arguments.output_dir.resolve(), arguments.compose_project, arguments.docker_executable)
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]), "error": report.get("error")}, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
