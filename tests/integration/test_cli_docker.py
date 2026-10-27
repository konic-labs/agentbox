"""CLI + HTTP server against Docker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentbox.cli.main import app

pytestmark = pytest.mark.docker
runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _need_docker() -> None:
    if not _docker_available():
        pytest.skip("Docker daemon not available")


def test_doctor_json() -> None:
    r = runner.invoke(app, ["--json", "doctor", "--no-probe"])
    data = json.loads(r.stdout)
    assert r.exit_code == 0 and data["docker"]["ok"] and "integrations" in data


def test_run_mock_resume_and_manifest(tmp_path: Path) -> None:
    out = tmp_path / "run"
    task = ROOT / "examples" / "tasks" / "fix_fizzbuzz" / "task.json"
    args = ["--json", "run", str(task), "--mock", "-n", "2", "-o", str(out), "--no-progress"]
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.stdout
    data = json.loads(r.stdout)
    assert data["n"] == 2 and (out / "run.json").exists() and (out / "summary.json").exists()
    r = runner.invoke(app, [*args[:5], "3", *args[6:], "--resume"])
    data = json.loads(r.stdout)
    assert data["n"] == 1 and data["skipped"] == 2
    lines = (out / "trajectories.jsonl").read_text().splitlines()
    assert len(lines) == 3
    r = runner.invoke(app, [*args, "--resume", "--max-steps", "3"])
    assert r.exit_code == 2  # config changed without --force


def test_bench_create_and_mock_run(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    r = runner.invoke(
        app,
        [
            "bench",
            "create",
            str(suite),
            "--from-tasks",
            str(ROOT / "examples" / "benchmarks" / "coding-mini" / "tasks"),
            "--suite-id",
            "mini",
            "--name",
            "mini",
        ],
    )
    assert r.exit_code == 0, r.stdout
    assert runner.invoke(app, ["bench", "validate", str(suite), "--strict"]).exit_code == 0
    out = tmp_path / "res"
    r = runner.invoke(
        app, ["--json", "bench", "run", str(suite), "--mock", "-o", str(out), "--no-progress"]
    )
    assert r.exit_code == 0, r.stdout
    assert (out / "report.json").exists() and (out / "models" / "mock" / "trajectories").is_dir()
    r = runner.invoke(app, ["bench", "show", str(out)])
    assert "Benchmark Report" in r.stdout


def test_bench_report_paths_survive_resume_and_moves(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    tasks = ROOT / "examples" / "benchmarks" / "coding-mini" / "tasks"
    create = ["bench", "create", str(suite), "--from-tasks", str(tasks)]
    assert runner.invoke(app, [*create, "--suite-id", "mini", "--name", "mini"]).exit_code == 0
    out = tmp_path / "res"
    run = ["--json", "bench", "run", str(suite), "--mock", "-o", str(out), "--no-progress"]
    assert runner.invoke(app, [*run, "--limit", "2"]).exit_code == 0
    r = runner.invoke(app, [*run, "--limit", "2", "--resume"])
    assert r.exit_code == 0, r.stdout
    moved = tmp_path / "archive" / "res"
    moved.parent.mkdir()
    out.rename(moved)
    by_task = json.loads((moved / "report.json").read_text())["models"][0]["by_task"]
    assert len(by_task) == 2
    for task in by_task:
        # every rollout, resumed or not, is listed relative to the report
        assert len(task["trajectory_paths"]) == task["n"] > 0
        for path in task["trajectory_paths"]:
            assert not Path(path).is_absolute() and (moved / path).is_file()


def test_sandbox_lifecycle_cli() -> None:
    task = ROOT / "examples" / "tasks" / "fix_fizzbuzz" / "task.json"
    r = runner.invoke(app, ["--json", "sandbox", "up", str(task)])
    assert r.exit_code == 0, r.stdout
    name = json.loads(r.stdout)["name"]
    try:
        r = runner.invoke(app, ["sandbox", "exec", name, "--", "ls", "/workspace"])
        assert "fizzbuzz.py" in r.stdout
        r = runner.invoke(app, ["--json", "sandbox", "ls"])
        assert name in [row["name"] for row in json.loads(r.stdout)]
        r = runner.invoke(app, ["--json", "sandbox", "prune", "--dry-run"])
        assert r.exit_code == 0
    finally:
        runner.invoke(app, ["sandbox", "rm", name])


async def test_http_server_roundtrip() -> None:
    pytest.importorskip("fastapi")
    import httpx

    from agentbox.config import ResourceLimits, SandboxConfig
    from agentbox.env.remote import RemoteEnv
    from agentbox.server.app import ServerSettings, create_app
    from agentbox.tasks.loader import load_task

    app_ = create_app(
        ServerSettings(
            sandbox=SandboxConfig(limits=ResourceLimits(network_disabled=True)), token="s3cret"
        )
    )
    task = load_task(ROOT / "examples" / "tasks" / "fix_fizzbuzz" / "task.json")
    async with app_.router.lifespan_context(app_):
        transport = httpx.ASGITransport(app=app_)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as raw:
            assert (await raw.post("/v1/sessions", json={})).status_code == 401
        client = httpx.AsyncClient(
            transport=transport,
            base_url="http://t",
            headers={"authorization": "Bearer s3cret"},
            timeout=120,
        )
        env = RemoteEnv("http://t", client=client)
        obs = await env.reset(task)
        assert obs.tools and obs.messages[1]["role"] == "user"
        out = await env.call_tool("read_file", {"path": "fizzbuzz.py"})
        assert "def fizzbuzz" in out["content"]
        res = await env.step("I am done.")
        assert res.done
        ev = await env.evaluate()
        assert ev.success is False
        traj = await env.trajectory()
        assert traj.task_id == task.task_id
        await env.close_session()
        metrics = (await client.get("/metrics")).text
        assert "agentbox_sessions_created_total 1" in metrics
        await client.aclose()


def test_bench_create_freezes_the_requested_agent(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from agentbox.benchmark.loader import load_suite
    from agentbox.cli.main import app

    tasks = (
        Path(__file__).resolve().parents[2] / "examples" / "benchmarks" / "coding-mini" / "tasks"
    )
    args = ["bench", "create", str(tmp_path / "s"), "--from-tasks", str(tasks)]
    args += [
        "--suite-id",
        "s",
        "--name",
        "s",
        "--tools",
        "read_file,write_file",
        "--max-steps",
        "6",
    ]
    r = CliRunner().invoke(app, args, catch_exceptions=False)
    assert r.exit_code == 0, r.stdout
    agent = load_suite(tmp_path / "s").manifest.agent
    assert agent.tools == ["read_file", "write_file"] and agent.max_steps == 6
    assert agent.drop_tools_prob == 0.0
