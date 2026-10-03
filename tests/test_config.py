"""验证配置字段、边界值和 Embedding 索引身份的规范化。"""

import pytest

from app.config import Settings


def test_retrieval_policy_settings_are_read_without_changing_embedding_identity() -> None:
    """读取独立召回策略及门槛，切换召回不改变已有向量空间身份。"""

    settings = Settings.from_environment({
        "TRACERAG_RETRIEVAL_STRATEGY": " HYBRID ",
        "TRACERAG_BM25_MIN_SCORE": "0.5",
        "TRACERAG_RRF_K": "80",
    })
    assert settings.retrieval_strategy == "hybrid"
    assert settings.bm25_min_score == 0.5
    assert settings.rrf_k == 80
    assert settings.embedding_identity == Settings().embedding_identity


@pytest.mark.parametrize("name,value", [
    ("RETRIEVAL_STRATEGY", "unknown"), ("BM25_MIN_SCORE", "nan"),
    ("BM25_MIN_SCORE", "inf"), ("BM25_MIN_SCORE", "-1"),
    ("BM25_MIN_SCORE", "wrong"), ("RRF_K", "0"), ("RRF_K", "1.5"),
])
def test_invalid_retrieval_settings_fail_at_startup(name: str, value: str) -> None:
    """无效召回策略与门槛在启动时失败，避免查询期间才暴露错误。"""

    with pytest.raises(ValueError, match="TRACERAG_"):
        Settings.from_environment({f"TRACERAG_{name}": value})


def test_settings_read_prefixed_environment_values() -> None:
    """验证配置只读取带项目前缀的环境变量。"""
    settings = Settings.from_environment(
        {
            "TRACERAG_APP_NAME": "TraceRAG Test",
            "TRACERAG_ENVIRONMENT": "test",
            "TRACERAG_VERSION": "0.1.0-test",
            "TRACERAG_HOST": "0.0.0.0",
            "TRACERAG_PORT": "9000",
            "TRACERAG_LOG_LEVEL": "debug",
        }
    )

    assert settings.app_name == "TraceRAG Test"
    assert settings.environment == "test"
    assert settings.version == "0.1.0-test"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize("port", ["0", "65536", "not-a-number"])
def test_settings_reject_invalid_ports(port: str) -> None:
    """验证配置拒绝超出合法范围的服务端口。"""
    with pytest.raises(ValueError, match="TRACERAG_PORT"):
        Settings.from_environment({"TRACERAG_PORT": port})


def test_settings_embedding_identity_uses_local_model_path(tmp_path) -> None:
    """验证本地模型路径和内容指纹构成向量身份。"""
    model_path = tmp_path / "bge-model"
    model_path.mkdir()
    (model_path / "modules.json").write_text("[]", encoding="utf-8")
    (model_path / "weights.bin").write_bytes(b"test-weights-v1")
    settings = Settings.from_environment(
        {
            "TRACERAG_EMBEDDING_PROVIDER": "local",
            "TRACERAG_LOCAL_EMBEDDING_PATH": str(model_path),
        }
    )

    assert settings.embedding_identity.provider == "local"
    assert settings.embedding_identity.model == str(model_path.resolve()).casefold()
    assert len(settings.embedding_identity.model_fingerprint) == 64


def test_remote_embedding_identity_includes_credential_free_endpoint() -> None:
    """验证远程模型身份区分端点且不保存凭据。"""
    first = Settings(
        embedding_model="same-alias",
        base_url="https://user:secret@API.Example.com:443/v1/?token=secret#fragment",
    )
    same_service = Settings(
        embedding_model="same-alias", base_url="https://api.example.com/v1"
    )
    other_service = Settings(
        embedding_model="same-alias", base_url="https://another.example/v1"
    )

    assert first.embedding_identity == same_service.embedding_identity
    assert first.embedding_identity.endpoint == "https://api.example.com/v1"
    assert "secret" not in first.embedding_identity.endpoint
    assert first.embedding_identity != other_service.embedding_identity


def test_local_embedding_identity_changes_when_weights_change_in_place(tmp_path) -> None:
    """验证同一路径的模型权重变化会更新身份指纹。"""
    model_path = tmp_path / "bge-model"
    model_path.mkdir()
    (model_path / "modules.json").write_text("[]", encoding="utf-8")
    weights_path = model_path / "weights.bin"
    weights_path.write_bytes(b"same-path-weight-v1")
    settings = Settings(
        embedding_provider="local", local_embedding_path=str(model_path)
    )
    first_fingerprint = settings.embedding_identity.model_fingerprint

    weights_path.write_bytes(b"same-path-weight-v2")
    replaced_settings = Settings(
        embedding_provider="local", local_embedding_path=str(model_path)
    )

    assert replaced_settings.embedding_identity.model == settings.embedding_identity.model
    assert replaced_settings.embedding_identity.model_fingerprint != first_fingerprint


def test_local_embedding_identity_requires_a_model_path() -> None:
    """验证本地 Embedding 配置必须指定模型目录。"""
    with pytest.raises(ValueError, match="TRACERAG_LOCAL_EMBEDDING_PATH"):
        Settings(embedding_provider="local")
