import pytest

from app.config import Settings


def test_settings_read_prefixed_environment_values() -> None:
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
    with pytest.raises(ValueError, match="TRACERAG_PORT"):
        Settings.from_environment({"TRACERAG_PORT": port})
