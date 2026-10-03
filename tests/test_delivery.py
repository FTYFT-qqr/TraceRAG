"""验证发布版本一致性、Docker文件排除规则和固定协议的输入输出。"""

import json
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from app import __version__
from app.config import Settings
from app.main import create_app
from scripts.delivery_provider import app as provider


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent() -> None:
    """包元数据、默认配置、健康检查和示例配置共享同一个发布版本。"""
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == __version__ == Settings().version == "0.2.0"
    for example in (".env.example", ".env.docker.example"):
        assert "TRACERAG_VERSION=0.2.0" in (ROOT / example).read_text(encoding="utf-8")
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
