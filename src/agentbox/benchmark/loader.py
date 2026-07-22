"""Load and save benchmark suite directory packs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from agentbox.benchmark.schema import BenchmarkSuite, BenchmarkSuiteManifest
from agentbox.config import AgentConfig, SandboxConfig
from agentbox.errors import BenchmarkError, TaskValidationError
from agentbox.tasks.bulk import load_task_dataset
from agentbox.tasks.schema import Task
from agentbox.types import ToolMode


def load_suite(path: str | Path) -> BenchmarkSuite:
    """Load suite.json + tasks from a directory pack."""
    root = Path(path).resolve()
    manifest_path = root / "suite.json"
    if not manifest_path.is_file():
        raise BenchmarkError(f"suite.json not found in {root}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = BenchmarkSuiteManifest.model_validate(data)
    except Exception as exc:
        raise BenchmarkError(f"Invalid suite.json: {exc}") from exc

    tasks_dir = root / manifest.tasks_path
    if not tasks_dir.exists():
        raise BenchmarkError(f"tasks path not found: {tasks_dir}")

    tasks = load_task_dataset(tasks_dir)
    if not tasks:
        # try single-level task.json dirs only already covered; try direct json
        for p in sorted(tasks_dir.glob("*.json")):
            try:
                tasks.append(Task.from_json(p))
            except TaskValidationError:
                continue
    if not tasks:
        raise BenchmarkError(f"No tasks found under {tasks_dir}")

    # Fairness: benchmarks should not carry custom_tools blobs
    agent = manifest.agent.model_copy(update={"custom_tools": [], "drop_tools_prob": 0.0})
    manifest = manifest.model_copy(update={"agent": agent})
    return BenchmarkSuite(root=root, manifest=manifest, tasks=tasks)


def save_suite(suite: BenchmarkSuite, path: str | Path) -> Path:
    """Write suite pack to directory (suite.json + tasks/)."""
    root = Path(path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tasks_dir = root / suite.manifest.tasks_path
    tasks_dir.mkdir(parents=True, exist_ok=True)

    for task in suite.tasks:
        task_dir = tasks_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        # Write starter files into files/ and strip from task.json for cleanliness
        files_dir = task_dir / "files"
        if task.starter_files:
            files_dir.mkdir(parents=True, exist_ok=True)
            for rel, content in task.starter_files.items():
                fp = files_dir / rel
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
        slim = task.model_copy(update={"starter_files": {}})
        slim.save_json(task_dir / "task.json")

    # Serialize agent without custom_tools
    m = suite.manifest.model_copy(deep=True)
    m.agent = m.agent.model_copy(update={"custom_tools": []})
    (root / "suite.json").write_text(
        m.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return root


def create_suite_from_tasks(
    tasks_src: str | Path,
    dest: str | Path,
    *,
    suite_id: str,
    name: str,
    version: str = "1.0.0",
    description: str = "",
    sandbox: SandboxConfig | None = None,
    agent: AgentConfig | None = None,
    n_per_task: int = 1,
    concurrency: int = 8,
    freeze: bool = True,
) -> BenchmarkSuite:
    """Create a suite directory from a tasks folder or dataset."""
    src = Path(tasks_src)
    tasks = load_task_dataset(src)
    if not tasks and src.is_dir():
        for p in sorted(src.rglob("task.json")):
            try:
                tasks.append(Task.from_dir(p.parent))
            except TaskValidationError:
                continue
        for p in sorted(src.glob("*.json")):
            try:
                tasks.append(Task.from_json(p))
            except TaskValidationError:
                continue
    # de-dupe
    seen: set[str] = set()
    unique: list[Task] = []
    for t in tasks:
        if t.task_id in seen:
            continue
        seen.add(t.task_id)
        unique.append(t)
    if not unique:
        raise BenchmarkError(f"No tasks found in {src}")

    manifest = BenchmarkSuiteManifest(
        suite_id=suite_id,
        name=name,
        description=description,
        version=version,
        sandbox=sandbox or SandboxConfig(),
        agent=agent
        or AgentConfig(
            tools=ToolMode.STRUCTURED,
            max_steps=40,
            drop_tools_prob=0.0,
            custom_tools=[],
        ),
        n_per_task=n_per_task,
        concurrency=concurrency,
    )
    suite = BenchmarkSuite(manifest=manifest, tasks=unique)
    if freeze:
        suite = suite.freeze()
    save_suite(suite, dest)
    suite.root = Path(dest).resolve()
    return suite


def snapshot_suite(suite: BenchmarkSuite, dest: str | Path) -> Path:
    """Copy frozen suite into results for audit."""
    dest_p = Path(dest)
    if suite.root and suite.root.is_dir():
        if dest_p.exists():
            shutil.rmtree(dest_p)
        shutil.copytree(suite.root, dest_p)
        return dest_p
    return save_suite(suite, dest_p)
