"""验证配置只从项目当前工作目录加载 .env。"""

from app.config import get_settings


def test_get_settings_loads_dotenv_from_project_working_directory(monkeypatch, tmp_path) -> None:
    """切换临时项目目录，避免依赖开发者机器上的真实密钥文件。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRACERAG_API_KEY", raising=False)
    (tmp_path / ".env").write_text("TRACERAG_API_KEY=project-local-key\n", encoding="utf-8")
    get_settings.cache_clear()

    try:
        settings = get_settings()
    finally:
        get_settings.cache_clear()

    assert settings.api_key == "project-local-key"
