"""验证 Embedding、Chat、代理和索引配置的读取行为。"""

from app.config import Settings


def test_settings_read_model_and_api_configuration() -> None:
    """验证环境配置能够正确读取模型、API 地址和密钥字段。"""
    settings = Settings.from_environment(
        {
            "TRACERAG_API_KEY": "local-test-key",
            "TRACERAG_BASE_URL": "https://gateway.example/v1",
            "TRACERAG_API_PROXY_URL": "http://127.0.0.1:7897",
            "TRACERAG_EMBEDDING_MODEL": "embed-model",
            "TRACERAG_CHAT_MODEL": "chat-model",
            "TRACERAG_INDEX_DIR": "tmp/index",
        }
    )

    assert settings.api_key == "local-test-key"
    assert settings.base_url == "https://gateway.example/v1"
    assert settings.api_proxy_url == "http://127.0.0.1:7897"
    assert settings.embedding_model == "embed-model"
    assert settings.chat_model == "chat-model"
    assert settings.index_dir == "tmp/index"


def test_settings_fall_back_to_openai_api_key() -> None:
    """验证专用密钥缺失时可以回退到兼容的环境变量。"""
    settings = Settings.from_environment({"OPENAI_API_KEY": "openai-key"})

    assert settings.api_key == "openai-key"
