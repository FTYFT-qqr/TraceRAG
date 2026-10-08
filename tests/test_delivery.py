"""验证发布版本一致性、Docker文件排除规则和固定协议的输入输出。"""

import hashlib
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from app import __version__
from app.config import Settings
from app.main import create_app
from scripts.delivery_provider import app as provider
from scripts.delivery_smoke import source_identity, stop


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent() -> None:
    """包元数据、默认配置、健康检查和示例配置共享同一个发布版本。"""
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == __version__ == Settings().version == "0.2.1"
    for example in (".env.example", ".env.docker.example"):
        assert "TRACERAG_VERSION=0.2.1" in (ROOT / example).read_text(encoding="utf-8")
    with TestClient(create_app(Settings())) as client:
        assert client.get("/health").json()["version"] == __version__


def test_fixed_provider_only_selects_existing_evidence_ids() -> None:
    """固定工程服务使用原有短句编号，不伪造自由回答或来源。"""
    content = {"question": "工程演示设备最长可以借用多久？", "evidence": [{"passages": [{"id": "2:3", "text": "每次借用最长4小时"}]}]}
    with TestClient(provider) as client:
        response = client.post("/v1/chat/completions", json={"model": "fixed", "messages": [{"content": json.dumps(content)}]})
    selected = json.loads(response.json()["choices"][0]["message"]["content"])
    assert selected == {"refused": False, "passage_ids": ["2:3"]}


def test_fixed_provider_refuses_missing_value_and_rejects_unlisted_question() -> None:
    """未知具体值正确拒答，非演示问题不能意外成为可用模型服务。"""
    with TestClient(provider) as client:
        content = {"question": "工程演示设备的购买价格是多少？", "evidence": []}
        response = client.post("/v1/chat/completions", json={"model": "fixed", "messages": [{"content": json.dumps(content)}]})
        assert json.loads(response.json()["choices"][0]["message"]["content"])["refused"] is True
        content["question"] = "任意其他问题"
        assert client.post("/v1/chat/completions", json={"model": "fixed", "messages": [{"content": json.dumps(content)}]}).status_code == 400


def test_container_context_uses_source_allowlist() -> None:
    """容器只纳入构建所需源码，密钥、索引和大报告默认排除。"""
    patterns = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "**" in patterns
    # 配置示例可进入镜像，真实凭据文件和目录仍不能被加入白名单。
    exceptions = {"!.env.example", "!.env.docker.example"}
    assert not any(line.startswith("!") and line not in exceptions and any(word in line for word in (".env", "indexes", "docs", ".planning")) for line in patterns)
    assert "127.0.0.1:" in (ROOT / "compose.yaml").read_text(encoding="utf-8")


def test_delivery_cli_handles_legacy_windows_output_encoding() -> None:
    """旧控制台编码不能让中文帮助或验收结果导致进程退出失败。"""
    environment = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/delivery_smoke.py"), "--help"],
        env=environment, capture_output=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert "新的独立报告目录" in completed.stdout.decode("utf-8")


def test_source_identity_without_git_keeps_file_fingerprints(monkeypatch) -> None:
    """精简容器缺少 Git 时仍用实际源码哈希标识本次验收代码。"""
    def missing_git(*args, **kwargs):
        """模拟精简镜像没有 Git 可执行文件的实际故障。"""
        raise FileNotFoundError("git")

    monkeypatch.setattr("scripts.delivery_smoke.subprocess.run", missing_git)
    identity = source_identity()
    assert identity["git_head"] is None
    assert identity["source_file_count"] == len(identity["files_sha256"])
    assert "scripts/evaluate_v02.py" in identity["files_sha256"]
    assert "scripts/delivery_smoke.py" in identity["files_sha256"]
    assert identity["source_sha256"] == hashlib.sha256(
        json.dumps(identity["files_sha256"], sort_keys=True).encode("utf-8")
    ).hexdigest()


def test_delivery_stop_releases_child_working_directory(tmp_path: Path) -> None:
    """停止验收进程后释放其工作目录，覆盖Windows虚拟环境启动器子进程。"""
    work = tmp_path / "delivery-process"
    work.mkdir()
    process = subprocess.Popen(
        [sys.executable, "-c", "from pathlib import Path; import time; Path('ready').write_text('ready'); time.sleep(60)"],
        cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        deadline = time.monotonic() + 15
        while not (work / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (work / "ready").exists(), "子进程必须实际启动后再检查清理"
        stop(process)
        assert process.poll() is not None
        (work / "ready").unlink()
        work.rmdir()
    finally:
        stop(process)
