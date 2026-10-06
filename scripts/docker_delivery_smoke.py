"""用固定协议服务检查候选镜像的源码、容器网络与空快照重启。"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

from app import __version__


ROOT = Path(__file__).resolve().parents[1]


def docker_call(executable: str, *arguments: str, timeout: int = 120) -> str:
    """以参数数组调用 Docker，只返回退出成功的标准输出。"""
    completed = subprocess.run(
        [executable, *arguments], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Docker {arguments[0]} 失败，退出码 {completed.returncode}: "
            f"{completed.stderr.strip()[-1000:]}"
        )
    return completed.stdout.strip()


def available_port() -> int:
    """为隔离验收容器选择仅绑定宿主回环地址的空闲端口。"""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def request_json(method: str, url: str, **kwargs: object) -> dict:
    """调用真实 HTTP 接口，并禁用宿主代理对本机请求的接管。"""
    with requests.Session() as session:
        session.trust_env = False
        response = session.request(method, url, timeout=40, **kwargs)
        response.raise_for_status()
        return response.json()


def wait_ready(url: str, *, expect_json: bool = True) -> dict | str:
    """等待容器端口实际可用，超时后交由外层保存容器日志。"""
    deadline = time.monotonic() + 150
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(url, timeout=3)
                response.raise_for_status()
                return response.json() if expect_json else response.text.strip()
        except (requests.RequestException, ValueError) as exc:
            last_error = type(exc).__name__
        time.sleep(0.5)
    raise TimeoutError(f"容器端口未就绪：{url}（最后错误：{last_error}）")


def compare_source(executable: str, container: str) -> dict:
    """逐文件比较 Dockerfile 实际复制的源码、测试、样本与构建元数据。"""
    paths = [
        "pyproject.toml", "requirements-release.txt", "README.md", ".env.example",
        ".env.docker.example", ".dockerignore", "compose.yaml", "compose.local.yaml",
    ]
    for directory in ("app", "scripts", "tests", "samples"):
        paths.extend(
            path.relative_to(ROOT).as_posix()
            for path in sorted((ROOT / directory).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    host = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    program = (
        "import hashlib,json,pathlib,sys; root=pathlib.Path('/opt/tracerag'); "
        "print(json.dumps({p: hashlib.sha256((root/p).read_bytes()).hexdigest() "
        "for p in json.loads(sys.argv[1])}, sort_keys=True))"
    )
    image = json.loads(docker_call(executable, "exec", container, "python", "-c", program, json.dumps(paths)))
    return {
        "host_sha256": hashlib.sha256(json.dumps(host, sort_keys=True).encode()).hexdigest(),
        "image_sha256": hashlib.sha256(json.dumps(image, sort_keys=True).encode()).hexdigest(),
        "file_count": len(paths),
        "mismatches": {name: {"host": host[name], "image": image.get(name)} for name in paths if host[name] != image.get(name)},
        "files_sha256": host,
    }


def run(executable: str, image: str, output: Path) -> dict:
    """创建一次性容器资源，检查联网、删除、重启和恢复后清理。"""
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "容器工程验收.json"
    if report_path.exists():
        raise FileExistsError(f"已有容器验收报告，不覆盖：{report_path}")
    suffix = uuid.uuid4().hex[:10]
    names = {
        "network": f"tracerag007-net-{suffix}",
        "volume": f"tracerag007-index-{suffix}",
        "provider": f"tracerag007-provider-{suffix}",
        "api": f"tracerag007-api-{suffix}",
        "ui": f"tracerag007-ui-{suffix}",
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, timeout=10, check=False,
    )
    report: dict = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "mode": "docker-offline-fixed-protocol",
        "scope": "候选镜像的源码、容器网络、API/UI、删除和空快照重启；固定模型不代表真实模型质量。",
        "image_tag": image, "resource_names": names,
        "git_head_at_check": revision.stdout.strip() if revision.returncode == 0 else None,
        "dockerfile_sha256": hashlib.sha256((ROOT / "Dockerfile").read_bytes()).hexdigest(),
        "checks": [], "passed": False,
    }
    created: list[tuple[str, str]] = []

    def save() -> None:
        """每项完成后落盘，使异常和清理状态可由机器复核。"""
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def record(name: str, passed: bool, detail: object) -> None:
        """保存明确的通过条件，失败时停止后续依赖检查。"""
        report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        save()
        if not passed:
            raise AssertionError(f"容器工程验收失败：{name}")

    def upload(base: str, content: bytes) -> dict:
        """只向本轮容器的独立索引卷导入公开演示样本。"""
        return request_json("POST", f"{base}/documents", files={"file": ("工程演示设备.txt", content, "text/plain")})

    def query(base: str) -> dict:
        """读取固定问题的真实 API 响应，核对检索是否为空。"""
        return request_json("POST", f"{base}/query", json={"query": "工程演示设备最长可以借用多久？", "top_k": 5})

    try:
        report["docker_version"] = docker_call(executable, "version", "--format", "{{.Server.Version}}")
        report["image_id"] = docker_call(executable, "image", "inspect", image, "--format", "{{.Id}}")
        save()
        docker_call(executable, "network", "create", names["network"])
        created.append(("network", names["network"]))
        docker_call(executable, "volume", "create", names["volume"])
        created.append(("volume", names["volume"]))
        docker_call(
            executable, "run", "-d", "--name", names["provider"],
            "--network", names["network"], "--network-alias", "provider", image,
            "python", "-m", "uvicorn", "delivery_provider:app", "--app-dir", "/opt/tracerag/scripts",
            "--host", "0.0.0.0", "--port", "8001",
        )
        created.append(("container", names["provider"]))
        api_port, ui_port = available_port(), available_port()
        docker_call(
            executable, "run", "-d", "--name", names["api"],
            "--network", names["network"], "--network-alias", "api",
            "-p", f"127.0.0.1:{api_port}:8000",
            "-v", f"{names['volume']}:/var/lib/tracerag",
            "-e", "TRACERAG_API_KEY=offline-delivery-placeholder",
            "-e", "TRACERAG_BASE_URL=http://provider:8001/v1",
            "-e", "TRACERAG_EMBEDDING_PROVIDER=openai",
            "-e", "TRACERAG_EMBEDDING_MODEL=delivery-fixed-embedding",
            "-e", "TRACERAG_CHAT_MODEL=delivery-fixed-chat",
            "-e", "TRACERAG_RETRIEVAL_STRATEGY=hybrid",
            "-e", "TRACERAG_INDEX_DIR=/var/lib/tracerag/indexes",
            image,
        )
        created.append(("container", names["api"]))
        docker_call(
            executable, "run", "-d", "--name", names["ui"],
            "--network", names["network"], "-p", f"127.0.0.1:{ui_port}:8501",
            "-e", "TRACERAG_API_URL=http://api:8000", image,
            "python", "-m", "streamlit", "run", "/opt/tracerag/app/ui.py",
            "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true",
            "--browser.gatherUsageStats=false",
        )
        created.append(("container", names["ui"]))
        base = f"http://127.0.0.1:{api_port}"
        health = wait_ready(f"{base}/health")
        ui_health = wait_ready(f"http://127.0.0.1:{ui_port}/_stcore/health", expect_json=False)
        record("API与UI健康", health["status"] == "ok" and health["version"] == __version__ and ui_health == "ok", {"api": health, "ui": ui_health})
        source = compare_source(executable, names["api"])
        record("镜像源码与当前工作区一致", not source["mismatches"], source)
        content = (ROOT / "samples" / "demo" / "工程演示设备.txt").read_bytes()
        first = upload(base, content)
        record("容器网络上传与持久化", first["chunk_count"] > 0 and not first["already_indexed"], first)
        answer = query(base)
        record("跨容器固定协议问答", not answer["rejected"] and "4小时" in answer["answer"] and answer["citations"][0]["file_name"] == "工程演示设备.txt", answer)
        docker_call(executable, "restart", names["api"])
        wait_ready(f"{base}/health")
        catalog = request_json("GET", f"{base}/documents")
        record("重启恢复非空快照", catalog["document_count"] == 1 and catalog["index_generation"] == first["index_generation"], catalog)
        deleted = request_json("DELETE", f"{base}/documents/{first['document_id']}")
        record("删除最后文档写入空快照", deleted["document_count"] == 0 and deleted["chunk_count"] == 0 and deleted["index_generation"] != first["index_generation"], deleted)
        empty = query(base)
        record("删除后旧内容不可检索", empty["rejected"] and not empty["citations"] and not empty["retrievals"], empty)
        docker_call(executable, "restart", names["api"])
        wait_ready(f"{base}/health")
        catalog = request_json("GET", f"{base}/documents")
        empty = query(base)
        record("重启后空快照仍生效", catalog["document_count"] == 0 and catalog["index_generation"] == deleted["index_generation"] and empty["rejected"] and not empty["retrievals"], {"catalog": catalog, "query": empty})
        again = upload(base, content)
        record("空快照可再次上传", again["chunk_count"] > 0 and not again["already_indexed"] and again["index_generation"] != deleted["index_generation"], again)
        answer = query(base)
        record("再次上传恢复问答", not answer["rejected"] and "4小时" in answer["answer"], answer)
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_errors = []
        for kind, name in reversed(created):
            if kind == "container":
                try:
                    (output / f"{kind}_{name}.log").write_text(
                        docker_call(executable, "logs", name), encoding="utf-8",
                    )
                except Exception as exc:
                    cleanup_errors.append(f"日志 {name}: {exc}")
            action = {"container": ("rm", "-f"), "volume": ("volume", "rm"), "network": ("network", "rm")}[kind]
            try:
                docker_call(executable, *action, name)
            except Exception as exc:
                cleanup_errors.append(f"清理 {name}: {exc}")
        report["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            report["passed"] = False
        save()
    return report


def main() -> None:
    """解析候选镜像和独立报告路径，以验收结果决定退出码。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.docker_executable, arguments.image, arguments.output_dir.resolve())
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]), "error": report.get("error"), "image_id": report.get("image_id")}, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
