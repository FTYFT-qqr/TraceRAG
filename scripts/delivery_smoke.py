"""在隔离目录启动真实 API/UI，使用本地固定协议服务检查工程交付链路。"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    """向操作系统申请本机空闲端口，避免占用用户现有的 API 和 UI。"""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_ready(url: str, process: subprocess.Popen, timeout: float = 45) -> dict | str:
    """在进程存活且健康检查成功后继续，否则给出明确的启动错误。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"子进程提前退出：{process.returncode}，请查看对应日志。")
        try:
            response = requests.get(url, timeout=2)
            if response.ok:
                try:
                    return response.json()
                except requests.exceptions.JSONDecodeError:
                    return response.text
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"等待服务超时：{url}")


def stop(process: subprocess.Popen) -> None:
    """仅停止本脚本创建的子进程，等待退出后再释放日志和临时目录。"""
    if process.poll() is None:
        if os.name == "nt":
            # Windows venv启动器会再创建Python子进程，必须停止本轮PID的整个子树。
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=20, check=False,
            )
            if completed.returncode and process.poll() is None:
                raise RuntimeError("停止验收进程树失败，不能继续清理临时目录。")
            process.wait(timeout=10)
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run(output: Path) -> dict:
    """检查安装后启动、上传、重复跳过、替换、回答、拒答、UI及重启恢复。"""
    output.mkdir(parents=True, exist_ok=False)
    processes = []
    handles = []
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline-fixed-protocol",
        "scope": "真实应用 HTTP 链路及索引持久化；固定模型协议不作为真实模型质量结论。",
        "python": sys.version, "checks": [], "passed": False,
    }

    def record(name: str, condition: bool, detail: object) -> None:
        """保存单项结果并立即中止失败链路，避免生成虚假的通过报告。"""
        report["checks"].append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"工程验收失败：{name}")

    with tempfile.TemporaryDirectory(prefix="tracerag-delivery-") as work:
        work_dir = Path(work)
        provider_port, api_port, ui_port = free_port(), free_port(), free_port()
        # 清除用户模型配置，防止隔离验收访问真实密钥、外部模型或应用索引。
        environment = {key: value for key, value in os.environ.items() if not key.startswith(("TRACERAG_", "OPENAI_"))}
        environment.update({
            "TRACERAG_API_KEY": "offline-delivery-placeholder",
            "TRACERAG_BASE_URL": f"http://127.0.0.1:{provider_port}/v1",
            "TRACERAG_EMBEDDING_PROVIDER": "openai",
            "TRACERAG_EMBEDDING_MODEL": "delivery-fixed-embedding",
            "TRACERAG_CHAT_MODEL": "delivery-fixed-chat",
            "TRACERAG_RETRIEVAL_STRATEGY": "hybrid",
            "TRACERAG_INDEX_DIR": str(work_dir / "indexes"),
            "TRACERAG_ENVIRONMENT": "delivery-check",
            "TRACERAG_API_URL": f"http://127.0.0.1:{api_port}",
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
            "USERPROFILE": str(work_dir), "HOME": str(work_dir),
        })

        def start(name: str, arguments: list[str]) -> subprocess.Popen:
            """创建隐藏的验收服务，并把输出保存到本轮独立目录。"""
            log = (output / f"{name}.log").open("w", encoding="utf-8")
            handles.append(log)
            process = subprocess.Popen(
                [sys.executable, *arguments], cwd=work_dir, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            processes.append(process)
            return process

        api_arguments = ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(api_port)]
        try:
            provider = start("固定协议服务", ["-m", "uvicorn", "delivery_provider:app", "--app-dir", str(ROOT / "scripts"), "--host", "127.0.0.1", "--port", str(provider_port)])
            wait_ready(f"http://127.0.0.1:{provider_port}/openapi.json", provider)
            api = start("接口启动", api_arguments)
            health = wait_ready(f"http://127.0.0.1:{api_port}/health", api)
            record("API版本和健康检查", health["version"] == "0.2.0" and health["status"] == "ok", health)
            ui = start("界面启动", ["-m", "streamlit", "run", str(ROOT / "app" / "ui.py"), "--server.address=127.0.0.1", f"--server.port={ui_port}", "--server.headless=true", "--browser.gatherUsageStats=false"])
            ui_health = wait_ready(f"http://127.0.0.1:{ui_port}/_stcore/health", ui)
            record("UI启动", ui_health == "ok", ui_health)
            base = f"http://127.0.0.1:{api_port}"
            content = (ROOT / "samples" / "demo" / "工程演示设备.txt").read_bytes()

            def upload(data: bytes) -> dict:
                """通过实际 multipart 接口上传固定样例并要求 HTTP 成功。"""
                response = requests.post(f"{base}/documents", files={"file": ("工程演示设备.txt", data, "text/plain")}, timeout=30)
                response.raise_for_status()
                return response.json()

            def query(question: str) -> dict:
                """通过实际查询接口取得包含引用和召回分的正式回答。"""
                response = requests.post(f"{base}/query", json={"query": question, "top_k": 5}, timeout=30)
                response.raise_for_status()
                return response.json()

            first = upload(content)
            record("首次上传", first["chunk_count"] > 0 and not first["already_indexed"], first)
            duplicate = upload(content)
            record("重复上传跳过", duplicate["already_indexed"] and duplicate["index_generation"] == first["index_generation"], duplicate)
            supported = query("工程演示设备最长可以借用多久？")
            record("有证据回答和来源", not supported["rejected"] and "4小时" in supported["answer"] and supported["citations"][0]["file_name"] == "工程演示设备.txt" and supported["retrievals"][0]["score_kind"] == "rrf", supported)
            unsupported = query("工程演示设备的购买价格是多少？")
            record("无证据拒答", unsupported["rejected"] and not unsupported["citations"], unsupported)
            replacement = upload(content + "\n最新登记规则要求登记姓名。".encode("utf-8"))
            record("同名替换", replacement["replaced_existing"] and replacement["index_generation"] != first["index_generation"], replacement)
            stop(api)
            restored = start("接口重启", api_arguments)
            wait_ready(f"{base}/health", restored)
            recovered = query("工程演示设备最长可以借用多久？")
            record("重启恢复", not recovered["rejected"] and "4小时" in recovered["answer"] and any("最新登记规则" in hit["text"] for hit in recovered["retrievals"]), recovered)
            record("恢复后仍跳过重复上传", upload(content + "\n最新登记规则要求登记姓名。".encode("utf-8"))["already_indexed"], replacement["index_generation"])
            report["passed"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            for process in reversed(processes):
                stop(process)
            for handle in handles:
                handle.close()
    (output / "工程验收.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """解析输出目录并以真实验收结果决定命令退出码。"""
    # Windows托管runner的重定向输出可能使用cp1252，明确以UTF-8打印中文。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="新的独立报告目录，已有目录不会覆盖")
    arguments = parser.parse_args()
    report = run(arguments.output_dir.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
