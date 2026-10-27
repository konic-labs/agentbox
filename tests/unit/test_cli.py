"""CLI commands that need no Docker (typer CliRunner)."""

from __future__ import annotations

import json
from datetime import UTC
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


def test_unknown_config_keys_are_reported(tmp_path: Path) -> None:
    cfg = tmp_path / "agentbox.yaml"
    cfg.write_text("sandobx:\n  image: x\nmodels:\n  m: {model: x}\n")
    r = runner.invoke(app, ["--json", "--config", str(cfg), "config", "validate"])
    assert r.exit_code == 0 and json.loads(r.stdout)["unknown_keys"] == ["sandobx"]
    r = runner.invoke(app, ["--config", str(cfg), "config", "validate", "--strict"])
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
    r = runner.invoke(
        app, ["traj", "filter", str(src), "-o", str(tmp_path / "ok.jsonl"), "--min-reward", "1"]
    )
    assert r.exit_code == 0 and len((tmp_path / "ok.jsonl").read_text().splitlines()) == 1
    assert runner.invoke(app, ["traj", "validate", str(src)]).exit_code == 0
    r = runner.invoke(
        app, ["traj", "export", str(src), "-f", "grpo", "-o", str(tmp_path / "g.jsonl")]
    )
    assert r.exit_code == 0 and "grpo" in r.stdout
    out = tmp_path / "rloo.jsonl"
    r = runner.invoke(
        app, ["traj", "export", str(src), "-f", "grpo", "--advantage", "rloo", "-o", str(out)]
    )
    group = next(g for g in map(json.loads, out.read_text().splitlines()) if g["task_id"] == "a")
    assert r.exit_code == 0 and group["advantages"] == [1.0, -1.0]  # leave-one-out baseline
    r = runner.invoke(app, ["traj", "export", str(src), "-f", "grpo", "--advantage", "nope"])
    assert r.exit_code == 2
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
    from datetime import datetime

    from agentbox.benchmark.report import build_model_report
    from agentbox.benchmark.schema import BenchmarkReport

    def report(successes: list[bool]) -> Path:
        trajs = [_traj(f"t{i}", 0, 1.0 if ok else 0.0) for i, ok in enumerate(successes)]
        mr = build_model_report("m", {"model": "m"}, trajs, wall_clock_s=1.0)
        now = datetime.now(UTC)
        rep = BenchmarkReport(
            report_id="r",
            suite_id="s",
            suite_version="1",
            agentbox_version="x",
            started_at=now,
            finished_at=now,
            models=[mr],
        )
        path = tmp_path / f"r{len(list(tmp_path.iterdir()))}.json"
        rep.save_json(path)
        return path

    a = report([True] * 8 + [False] * 2)
    b = report([False] * 8 + [False] * 2)
    r = runner.invoke(
        app, ["--json", "bench", "compare", str(a), str(b), "--fail-on-regression", "0.1"]
    )
    assert r.exit_code == 1
    paired = json.loads(r.stdout)["deltas"][0]["paired"]
    assert paired["losses"] == 8 and paired["sign_test_p"] < 0.01


def test_reward_flag_parsing() -> None:
    from agentbox.cli.run import _reward_specs

    specs = _reward_specs(
        ["verifier", "step_penalty:0.5,per_step=0.02", "pkg.mod:fn", "pkg.mod:fn:2,flag=true"]
    )
    assert [(s.name, s.weight) for s in specs] == [
        ("verifier", 1.0),
        ("step_penalty", 0.5),
        ("pkg.mod:fn", 1.0),
        ("pkg.mod:fn", 2.0),
    ]
    assert specs[1].params == {"per_step": 0.02} and specs[3].params == {"flag": True}


def test_task_select_by_pass_rate(tmp_path: Path) -> None:
    from agentbox.tasks.schema import Task, VerifierSpec

    tasks = tmp_path / "tasks.jsonl"
    rows = [
        Task(task_id=tid, description="d", verifier=VerifierSpec(type="pytest"))
        for tid in ("a", "b", "c", "never_run")
    ]
    tasks.write_text("".join(t.model_dump_json() + "\n" for t in rows))
    # a: 1/2 solved (learnable), b: always solved, c: never solved
    runs = tmp_path / "runs.jsonl"
    write_jsonl(
        [_traj("a", 0, 1.0), _traj("a", 1, 0.0), _traj("b", 0, 1.0), _traj("c", 0, 0.0)], runs
    )
    out = tmp_path / "learnable.jsonl"
    args = ["task", "select", str(tasks), "--runs", str(runs), "-o", str(out)]
    r = runner.invoke(app, ["--json", *args, "--min-pass", "0.01", "--max-pass", "0.99"])
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["selected"] == 2  # a + the unseen task
    ids = [json.loads(line)["task_id"] for line in out.read_text().splitlines()]
    assert ids == ["a", "never_run"]
    r = runner.invoke(app, [*args, "--min-pass", "0.01", "--max-pass", "0.99", "--drop-unseen"])
    assert [json.loads(x)["task_id"] for x in out.read_text().splitlines()] == ["a"]


def test_teacher_profile_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from agentbox.cli.generate import _teacher
    from agentbox.config_load import EndpointConfig, ProjectConfig

    monkeypatch.setenv("TEACHER_KEY", "sk-teacher")
    project = ProjectConfig(teacher=EndpointConfig(model="t", api_key_env="TEACHER_KEY"))
    st = SimpleNamespace(project=project)
    model, _, key, *_ = _teacher(st, None, None, None, None, None, None, None)  # type: ignore[arg-type]
    assert (model, key) == ("t", "sk-teacher")
    _, _, flag_key, *_ = _teacher(st, None, None, "sk-flag", None, None, None, None)  # type: ignore[arg-type]
    assert flag_key == "sk-flag"


async def test_task_generator_is_an_async_context_manager() -> None:
    from agentbox.tasks.generate import GenerateConfig, TaskGenerator

    async with TaskGenerator(GenerateConfig(model="m", use_dspy=False)) as gen:
        assert gen._client is None  # clients are created lazily
