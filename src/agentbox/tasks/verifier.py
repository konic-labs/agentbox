"""Run task verifiers inside the sandbox.

Order of operations (GH #49, #50)::

    1. tamper checks   — restore protected starter files, delete forbidden new
                          files (conftest.py & co.), record what was touched
    2. hidden files    — written now, never visible to the agent
    3. pre_commands    — e.g. install test-only dependencies
    4. verifier        — hardened env (no PYTEST_ADDOPTS, no user site, no cache)
    5. reward          — binary | pytest_fraction (JUnit XML) | stdout | custom
"""

from __future__ import annotations

import asyncio
import fnmatch
import importlib
import json
import logging
import posixpath
import re
import shlex
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from agentbox.errors import VerifierError
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import VerifierType

if TYPE_CHECKING:
    from agentbox.sandbox.manager import SandboxManager
    from agentbox.sandbox.types import ExecResult, Sandbox

logger = logging.getLogger("agentbox.verifier")

_HARDENED_ENV = {
    "PYTEST_ADDOPTS": "",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PY_COLORS": "0",
}
_SUMMARY_RE = {
    "passed": re.compile(r"(\d+) passed"),
    "failed": re.compile(r"(\d+) failed"),
    "errors": re.compile(r"(\d+) errors?\b"),
    "skipped": re.compile(r"(\d+) skipped"),
}
_PRUNE = (".git", "node_modules", ".venv", "__pycache__")


class VerifyResult(BaseModel):
    reward: float
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    command: str
    timed_out: bool = False
    tests_passed: int | None = None
    tests_failed: int | None = None
    tests_total: int | None = None
    tampered: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(posixpath.basename(path), p)
               for p in patterns)


def _is_pytest(command: str) -> bool:
    return bool(re.search(r"(^|[\s/])(py\.test|pytest)(\s|$)", command))


def parse_junit(xml_bytes: bytes) -> dict[str, int] | None:
    """Sum tests/failures/errors/skipped over a pytest JUnit XML report."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if not suites:
        return None
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for s in suites:
        for key in totals:
            try:
                totals[key] += int(s.get(key, "0"))
            except ValueError:
                pass
    return totals


def parse_pytest_summary(text: str) -> dict[str, int] | None:
    tail = text[-4000:]
    counts = {k: int(m.group(1)) for k, rx in _SUMMARY_RE.items() if (m := rx.search(tail))}
    if not counts:
        return None
    return {
        "tests": sum(counts.values()),
        "failures": counts.get("failed", 0),
        "errors": counts.get("errors", 0),
        "skipped": counts.get("skipped", 0),
    }


def parse_stdout_reward(stdout: str) -> tuple[float, dict[str, Any]] | None:
    """Last non-empty stdout line as a float or JSON ``{"reward": x, ...}``."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), {}
        if isinstance(value, dict) and isinstance(value.get("reward"), (int, float)):
            extra = {k: v for k, v in value.items() if k != "reward"}
            return float(value["reward"]), extra
        return None
    return None


def load_callable(path: str) -> Callable[..., Any]:
    """Import ``package.module:function`` (or ``package.module.function``)."""
    module_name, sep, attr = path.partition(":")
    if not sep:
        module_name, _, attr = path.rpartition(".")
    fn = getattr(importlib.import_module(module_name), attr)
    if not callable(fn):
        raise TypeError(f"{path} is not callable")
    return fn  # type: ignore[no-any-return]


class Verifier:
    def __init__(self, manager: SandboxManager) -> None:
        self.manager = manager

    def build_command(self, spec: VerifierSpec) -> str:
        if spec.type in (VerifierType.PYTEST, VerifierType.TEST):
            return spec.command or f"python -m pytest -q {spec.path or ''}".strip()
        if spec.command:
            return spec.command
        raise ValueError(f"verifier type {spec.type.value!r} needs a command")

    @staticmethod
    def pytest_addopts(spec: VerifierSpec, junit_path: str | None) -> str | None:
        """Flags injected via PYTEST_ADDOPTS (works however pytest is invoked)."""
        opts = []
        if spec.hardened:
            opts.append("-p no:cacheprovider")
        if junit_path:
            opts.append(f"--junitxml={shlex.quote(junit_path)}")
        return " ".join(opts) if opts else None

    async def check_tamper(self, sandbox: Sandbox, spec: VerifierSpec, task: Task | None) -> list[str]:
        """Restore protected files and remove forbidden new files; return touched paths."""
        mgr = self.manager
        tampered: list[str] = []
        baseline = task.seed_files() if task is not None else {}
        if spec.protected_paths:
            protected = {p: d for p, d in baseline.items() if _matches(p, spec.protected_paths)}
            for path, original in protected.items():
                try:
                    current = await mgr.read_bytes(sandbox, path, max_bytes=len(original) + 1)
                except FileNotFoundError:
                    current = None
                if current != original:
                    tampered.append(path)
            if protected:
                await mgr.write_files(sandbox, protected)
        if spec.forbid_new_files:
            names = sorted({posixpath.basename(p) for p in spec.forbid_new_files})
            prune = " -o ".join(f"-name {shlex.quote(d)}" for d in _PRUNE)
            match = " -o ".join(f"-name {shlex.quote(n)}" for n in names)
            ws = shlex.quote(sandbox.workspace_dir)
            script = f"find {ws} \\( {prune} \\) -prune -o -type f \\( {match} \\) -print"
            res = await mgr.exec(sandbox, ["/bin/sh", "-c", script], timeout_s=30.0)
            prefix = sandbox.workspace_dir.rstrip("/") + "/"
            offenders = []
            for line in res.stdout.splitlines():
                rel = line.strip()[len(prefix):] if line.startswith(prefix) else line.strip()
                if rel and rel not in baseline and _matches(rel, spec.forbid_new_files):
                    offenders.append(rel)
            if offenders:
                await mgr.exec(
                    sandbox, ["rm", "-f", "--", *[prefix + o for o in offenders]], timeout_s=30.0
                )
                tampered.extend(offenders)
        return tampered

    async def verify(
        self,
        sandbox: Sandbox,
        spec: VerifierSpec,
        task: Task | None = None,
    ) -> VerifyResult:
        tampered = await self.check_tamper(sandbox, spec, task) if spec.tamper_protection else []
        if spec.hidden_files:
            await self.manager.write_files(sandbox, spec.hidden_files)
        env = dict(spec.env)
        if spec.hardened:
            env = {**_HARDENED_ENV, **env}
        for cmd in spec.pre_commands:
            pre = await self.manager.exec(
                sandbox, cmd, timeout_s=spec.timeout_s, env=env, workdir=spec.workdir
            )
            if not pre.ok:
                raise VerifierError(
                    f"verifier pre_command failed ({cmd!r}, exit={pre.exit_code}): "
                    f"{(pre.stderr or pre.stdout).strip()[-500:]}"
                )
        junit_path = None
        if spec.reward_mode == "pytest_fraction":
            junit_path = f"/tmp/agentbox-junit-{uuid.uuid4().hex[:12]}.xml"
        command = self.build_command(spec)
        addopts = self.pytest_addopts(spec, junit_path)
        if addopts is not None:
            env["PYTEST_ADDOPTS"] = addopts  # also overrides anything the agent exported
        t0 = time.monotonic()
        result = await self.manager.exec(
            sandbox, command, timeout_s=spec.timeout_s, env=env, workdir=spec.workdir
        )
        duration = time.monotonic() - t0
        success = not result.timed_out and result.exit_code == spec.success_exit_code
        reward = spec.reward_success if success else spec.reward_failure
        counts: dict[str, int] | None = None
        details: dict[str, Any] = {}

        if spec.reward_mode == "pytest_fraction":
            if junit_path is not None:
                try:
                    raw = await self.manager.backend.read_file(
                        sandbox.id, junit_path, max_bytes=4 * 1024 * 1024
                    )
                    counts = parse_junit(raw.data)
                except (FileNotFoundError, Exception) as exc:  # noqa: BLE001
                    logger.debug("no junit report (%s); falling back to summary", exc)
            counts = counts or parse_pytest_summary(result.stdout)
            if counts and counts["tests"] - counts["skipped"] > 0:
                denom = counts["tests"] - counts["skipped"]
                passed = denom - counts["failures"] - counts["errors"]
                frac = max(0.0, min(1.0, passed / denom))
                reward = spec.reward_failure + (spec.reward_success - spec.reward_failure) * frac
        elif spec.reward_mode == "stdout":
            parsed = parse_stdout_reward(result.stdout)
            if parsed is not None:
                reward, details = parsed
        elif spec.reward_mode == "custom" and spec.reward_fn:
            value = load_callable(spec.reward_fn)(result=result, task=task, spec=spec)
            if asyncio.iscoroutine(value):
                value = await value
            reward = float(value)
        if spec.success_threshold is not None:
            success = not result.timed_out and reward >= spec.success_threshold
        if counts is None and _is_pytest(command):
            counts = parse_pytest_summary(result.stdout)

        if tampered:
            details["tampered"] = tampered
            if spec.on_tamper == "fail":
                success = False
                reward = spec.tamper_reward if spec.tamper_reward is not None else spec.reward_failure

        tests_total = counts["tests"] if counts else None
        tests_failed = (counts["failures"] + counts["errors"]) if counts else None
        tests_passed = (
            counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
            if counts
            else None
        )
        return VerifyResult(
            reward=reward,
            success=success,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_s=duration,
            command=command,
            timed_out=result.timed_out,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_total=tests_total,
            tampered=tampered,
            details=details,
        )


def exec_result_summary(result: ExecResult) -> str:  # pragma: no cover - debugging helper
    return f"exit={result.exit_code} timed_out={result.timed_out}"
