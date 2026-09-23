from app.config import get_settings


def test_get_settings_loads_dotenv_from_project_working_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRACERAG_API_KEY", raising=False)
    (tmp_path / ".env").write_text("TRACERAG_API_KEY=project-local-key\n", encoding="utf-8")
    get_settings.cache_clear()

    try:
        settings = get_settings()
    finally:
        get_settings.cache_clear()

    assert settings.api_key == "project-local-key"
