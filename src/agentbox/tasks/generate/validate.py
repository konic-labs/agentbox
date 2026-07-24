"""Validate generated tasks: schema + live Docker QC (+ helpers for LLM judge)."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from agentbox.config import SandboxConfig
from agentbox.sandbox.manager import SandboxManager
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.tasks.seeder import TaskSeeder
from agentbox.tasks.verifier import Verifier
from agentbox.types import VerifierType

logger = logging.getLogger("agentbox.generate.validate")


class ValidationReport(BaseModel):
    ok: bool
    task: Task | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    seed_ok: bool | None = None
    verifier_runs: bool | None = None
    verifier_fails_on_starter: bool | None = None
    # LLM judge (optional second API call; same teacher endpoint)
    llm_ok: bool | None = None
    llm_score: float | None = None
    llm_accept: bool | None = None
    llm_reasons: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


def parse_task_from_prediction(
    *,
    task_id: str,
    description: str,
    starter_files_json: str,
    setup_commands_json: str,
    verifier_json: str,
    metadata_json: str,
    generator_model: str | None = None,
) -> Task:
    """Parse model fields into a validated Task."""
    starter_files = json.loads(starter_files_json)
    if not isinstance(starter_files, dict):
        raise ValueError("starter_files_json must be a JSON object")
    # Models sometimes leak schema keys into starter_files
    _junk = {"setup_commands", "verifier", "metadata", "task_id", "description"}
    starter_files = {
        str(k): str(v)
        for k, v in starter_files.items()
        if str(k) not in _junk and not str(k).startswith("__")
    }
    if not starter_files:
        raise ValueError("starter_files empty after filtering junk keys")

    setup_commands = json.loads(setup_commands_json or "[]")
    if not isinstance(setup_commands, list):
        raise ValueError("setup_commands_json must be a JSON list")
    setup_commands = [str(c) for c in setup_commands]

    verifier_raw = json.loads(verifier_json)
    if not isinstance(verifier_raw, dict):
        raise ValueError("verifier_json must be a JSON object")
    if "type" in verifier_raw and isinstance(verifier_raw["type"], str):
        verifier_raw["type"] = VerifierType(verifier_raw["type"])
    verifier = VerifierSpec.model_validate(verifier_raw)

    metadata = json.loads(metadata_json or "{}")
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must be a JSON object")
    metadata = dict(metadata)
    metadata.setdefault("source", "dspy")
    if generator_model:
        metadata["generator_model"] = generator_model

    return Task(
        task_id=str(task_id).strip() or f"gen_{uuid.uuid4().hex[:8]}",
        description=str(description).strip(),
        starter_files=starter_files,
        setup_commands=setup_commands,
        verifier=verifier,
        metadata=metadata,
    )


async def validate_task_live(
    task: Task,
    *,
    sandbox_config: SandboxConfig | None = None,
    expect_fail_on_starter: bool = True,
) -> ValidationReport:
    """Schema is assumed valid; run seed + verifier smoke in Docker."""
    errors: list[str] = []
    warnings: list[str] = []
    cfg = sandbox_config or SandboxConfig(
        limits=SandboxConfig().limits.model_copy(update={"network_disabled": False}),
        ensure_pytest=True,
    )
    manager = SandboxManager(cfg)
    sandbox = None
    try:
        sandbox = await manager.create(task_id=task.task_id, run_id=str(uuid.uuid4()))
        seed = await TaskSeeder(manager).seed(sandbox, task)
        if not seed.ok:
            return ValidationReport(
                ok=False,
                task=task,
                errors=[seed.error or "seed failed"],
                seed_ok=False,
            )

        verify = await Verifier(manager).verify(sandbox, task.verifier)
        verifier_runs = True
        fails_on_starter = not verify.success
        if expect_fail_on_starter and verify.success:
            warnings.append(
                "verifier already passes on starter files (task may have no work)"
            )
            return ValidationReport(
                ok=False,
                task=task,
                errors=["verifier passes on unfixed starter"],
                warnings=warnings,
                seed_ok=True,
                verifier_runs=True,
                verifier_fails_on_starter=False,
                details={"verify_exit_code": verify.exit_code},
            )

        return ValidationReport(
            ok=True,
            task=task,
            warnings=warnings,
            seed_ok=True,
            verifier_runs=verifier_runs,
            verifier_fails_on_starter=fails_on_starter,
            details={"verify_exit_code": verify.exit_code},
        )
    except Exception as exc:
        logger.exception("live validation failed")
        return ValidationReport(ok=False, task=task, errors=[str(exc)])
    finally:
        if sandbox is not None:
            await manager.destroy(sandbox)


def task_payload_for_judge(task: Task, *, max_file_chars: int = 6000) -> dict[str, str]:
    """Serialize a Task into compact JSON strings for the LLM judge."""
    files: dict[str, str] = {}
    for path, content in (task.starter_files or {}).items():
        text = str(content)
        if len(text) > max_file_chars:
            text = text[:max_file_chars] + "\n...[truncated]..."
        files[str(path)] = text
    return {
        "task_id": task.task_id,
        "description": task.description,
        "starter_files_json": json.dumps(files, ensure_ascii=False),
        "setup_commands_json": json.dumps(list(task.setup_commands or [])),
        "verifier_json": json.dumps(task.verifier.model_dump(mode="json")),
    }
