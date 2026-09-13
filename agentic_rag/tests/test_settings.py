from scripts.settings import RagSettings


def test_settings_repr_never_exposes_together_key():
    settings = RagSettings(together_api_key="secret-value")

    assert "secret-value" not in repr(settings)


def test_relative_environment_paths_are_resolved_from_project(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("RAG_PROJECT_DIR", str(tmp_path))
    monkeypatch.delenv("RAG_DATA_DIR", raising=False)
    monkeypatch.setenv("QDRANT_PATH", "./vectors")

    settings = RagSettings.from_env(tmp_path / "missing.env")

    assert settings.project_dir == tmp_path
    assert settings.data_dir == tmp_path / "data"
    assert settings.qdrant_path == tmp_path / "vectors"
