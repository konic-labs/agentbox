"""CLI commands that need no Docker (typer CliRunner)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentbox.cli.main import app
from agentbox.config_load import ConfigError, interpolate, load_project_config
from agentbox.trajectory.io import write_jsonl
from tests.unit.test_export_tito import _traj

runner = CliRunner()


def invoke(*args: str, cwd: Path | None = None) -> object:
    result = runner.invoke(app, list(args), catch_exceptions=False)
    return result


def test_version_and_json() -> None:
    r = runner.invoke(app, ["--json", "version"])
    assert r.exit_code == 0 and json.loads(r.stdout)["version"]


def test_init_config_and_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENTBOX_CONFIG", raising=False)
    r = runner.invoke(app, ["init", "."])
    assert r.exit_code == 0 and (tmp_path / "agentbox.yaml").exists()
    r = runner.invoke(app, ["config", "validate"])
    assert r.exit_code == 0, r.stdout
    r = runner.invoke(app, ["--json", "config", "show"])
    data = json.loads(r.stdout)
    assert data["default_model"] == "local" and data["models"]["local"]["api_key"] == "***"
    cfg = load_project_config(tmp_path / "agentbox.yaml")
    assert cfg.model().model == "Qwen/Qwen3-1.7B"
    with pytest.raises(ConfigError):
        cfg.model("nope")


def test_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AB_SET", "x")
    monkeypatch.delenv("AB_UNSET", raising=False)
    assert interpolate("${AB_SET}-${AB_UNSET:-dflt}") == "x-dflt"
    with pytest.raises(ConfigError, match="needed"):
        interpolate("${AB_UNSET:?needed}")
    with pytest.raises(ConfigError):
        interpolate("${AB_UNSET}")


def test_bad_config_is_reported(tmp_path: Path) -> None:
    bad = tmp_path / "agentbox.yaml"
    bad.write_text("models:\n  m:\n    model: x\n    bogus_field: 1\n")
    r = runner.invoke(app, ["--config", str(bad), "config", "validate"])
    assert r.exit_code == 2


def test_task_commands(tmp_path: Path) -> None:
    r = runner.invoke(app, ["task", "new", str(tmp_path / "t1")])
    assert r.exit_code == 0 and (tmp_path / "t1" / "files" / "test_solution.py").exists()
    assert runner.invoke(app, ["task", "validate", str(tmp_path)]).exit_code == 0
    r = runner.invoke(app, ["--json", "task", "ls", str(tmp_path)])
    assert json.loads(r.stdout)[0]["task_id"] == "t1"
    r = runner.invoke(app, ["task", "schema"])
    assert "verifier" in json.loads(r.stdout)["properties"]
    (tmp_path / "broken.json").write_text('{"task_id": "x"}')
    assert runner.invoke(app, ["task", "validate", str(tmp_path)]).exit_code == 1


def test_traj_commands(tmp_path: Path) -> None:
    src = tmp_path / "trajs.jsonl"
    write_jsonl([_traj("a", 0, 1.0), _traj("a", 1, 0.0), _traj("b", 0, 0.0)], src)
    r = runner.invoke(app, ["--json", "traj", "stats", str(src)])
    stats = json.loads(r.stdout)
    assert stats["n"] == 3 and stats["groups"]["groups"] == 2
    r = runner.invoke(app, ["traj", "filter", str(src), "-o", str(tmp_path / "ok.jsonl"), "--min-reward", "1"])
    assert r.exit_code == 0 and len((tmp_path / "ok.jsonl").read_text().splitlines()) == 1
    assert runner.invoke(app, ["traj", "validate", str(src)]).exit_code == 0
    r = runner.invoke(app, ["traj", "export", str(src), "-f", "grpo", "-o", str(tmp_path / "g.jsonl")])
    assert r.exit_code == 0 and "grpo" in r.stdout
    r = runner.invoke(app, ["--json", "traj", "diff", str(src), str(src)])
    assert json.loads(r.stdout)["first_divergent_tool_call"] is None
    r = runner.invoke(app, ["traj", "render", str(src), "-o", str(tmp_path / "d.html")])
    assert r.exit_code == 0 and (tmp_path / "d.html").stat().st_size > 1000
    r = runner.invoke(app, ["traj", "show", str(src)])
    assert "reward=" in r.stdout


def test_traj_validate_flags_dangling_calls(tmp_path: Path) -> None:
    t = _traj("a", 0, 1.0)
    t.messages = t.messages[:3]  # assistant tool call without its tool message
    write_jsonl([t], tmp_path / "bad.jsonl")
    assert runner.invoke(app, ["traj", "validate", str(tmp_path / "bad.jsonl")]).exit_code == 1


def test_integrations_listing() -> None:
    r = runner.invoke(app, ["--json", "integrations"])
    names = {row["name"] for row in json.loads(r.stdout)}
    assert names == {"trl", "verl", "openrlhf", "skyrl", "art"}


def test_bench_compare_statistics(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    from agentbox.benchmark.report import build_model_report
    from agentbox.benchmark.schema import BenchmarkReport

    def report(successes: list[bool]) -> Path:
        trajs = [_traj(f"t{i}", 0, 1.0 if ok else 0.0) for i, ok in enumerate(successes)]
        mr = build_model_report("m", {"model": "m"}, trajs, wall_clock_s=1.0)
        now = datetime.now(timezone.utc)
        rep = BenchmarkReport(report_id="r", suite_id="s", suite_version="1", agentbox_version="x",
                              started_at=now, finished_at=now, models=[mr])
        path = tmp_path / f"r{len(list(tmp_path.iterdir()))}.json"
        rep.save_json(path)
        return path

    a = report([True] * 8 + [False] * 2)
    b = report([False] * 8 + [False] * 2)
    r = runner.invoke(app, ["--json", "bench", "compare", str(a), str(b), "--fail-on-regression", "0.1"])
    assert r.exit_code == 1
    paired = json.loads(r.stdout)["deltas"][0]["paired"]
    assert paired["losses"] == 8 and paired["sign_test_p"] < 0.01
