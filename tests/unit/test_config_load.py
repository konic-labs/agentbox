from pathlib import Path

from agentbox.config_load import ProjectConfig, load_project_config


def test_load_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENTBOX_CONFIG", raising=False)
    cfg = load_project_config()
    assert isinstance(cfg, ProjectConfig)
    assert cfg.generate.concurrency == 8


def test_load_yaml(tmp_path: Path, monkeypatch) -> None:
    try:
        import yaml  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("pyyaml not installed")
    p = tmp_path / "agentbox.yaml"
    p.write_text(
        """
teacher:
  model: m1
  base_url: http://localhost:8000/v1
  api_key: EMPTY
generate:
  concurrency: 4
  target: 10
students:
  - id: s1
    model: q
    base_url: http://localhost:11434/v1
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    cfg = load_project_config()
    assert cfg.teacher is not None
    assert cfg.teacher.model == "m1"
    assert cfg.generate.concurrency == 4
    assert cfg.students[0].id == "s1"
