"""Automated task generation via frontier models (+ optional DSPy)."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentbox.config import ModelConfig, SandboxConfig
from agentbox.errors import TaskGenerationError
from agentbox.tasks.generate.lm import build_openai_client, configure_dspy_lm
from agentbox.tasks.generate.validate import (
    ValidationReport,
    parse_task_from_prediction,
    validate_task_live,
)
from agentbox.tasks.schema import Task

logger = logging.getLogger("agentbox.generate")

Difficulty = Literal["easy", "medium", "hard"]


class GenerateConfig(BaseModel):
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.8
    max_tokens: int | None = 8192
    timeout_s: float = 180.0
    validate_in_docker: bool = True
    sandbox: SandboxConfig = Field(
        default_factory=lambda: SandboxConfig(
            limits=SandboxConfig().limits.model_copy(
                update={"network_disabled": False}
            ),
            ensure_pytest=True,
        )
    )
    max_retries: int = 2
    use_dspy: bool = True
    expect_fail_on_starter: bool = True


class TaskGenerator:
    """Generate complete Task definitions using a frontier model."""

    def __init__(self, config: GenerateConfig) -> None:
        self.config = config
        self.model_config = ModelConfig(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
        )
        self._dspy_module: Any | None = None
        if config.use_dspy:
            try:
                from agentbox.tasks.generate.signatures import get_generator_module

                configure_dspy_lm(self.model_config)
                self._dspy_module = get_generator_module()()
            except Exception as exc:
                logger.warning("DSPy unavailable (%s); falling back to OpenAI JSON", exc)
                self._dspy_module = None

    async def generate(
        self,
        *,
        difficulty: Difficulty = "easy",
        domain: str = "python",
        constraints: str | None = None,
        tags: list[str] | None = None,
    ) -> Task:
        last_error: str | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                task = await self._generate_once(
                    difficulty=difficulty,
                    domain=domain,
                    constraints=constraints or "",
                )
                if tags:
                    meta_tags = list(task.metadata.get("tags") or [])
                    task.metadata["tags"] = sorted(set(meta_tags + tags))

                if self.config.validate_in_docker:
                    report = await validate_task_live(
                        task,
                        sandbox_config=self.config.sandbox,
                        expect_fail_on_starter=self.config.expect_fail_on_starter,
                    )
                    if not report.ok:
                        last_error = "; ".join(report.errors) or "validation failed"
                        logger.warning(
                            "generate attempt %d failed QC: %s", attempt + 1, last_error
                        )
                        continue
                return task
            except Exception as exc:
                last_error = str(exc)
                logger.warning("generate attempt %d error: %s", attempt + 1, exc)

        raise TaskGenerationError(
            f"Failed to generate valid task after {self.config.max_retries + 1} attempts: "
            f"{last_error}"
        )

    async def generate_many(
        self,
        n: int,
        **kwargs: Any,
    ) -> list[Task]:
        tasks: list[Task] = []
        for i in range(n):
            try:
                task = await self.generate(**kwargs)
                # ensure unique ids
                if any(t.task_id == task.task_id for t in tasks):
                    task.task_id = f"{task.task_id}_{uuid.uuid4().hex[:6]}"
                tasks.append(task)
            except TaskGenerationError as exc:
                logger.error("generate_many item %d failed: %s", i, exc)
        return tasks

    async def validate_task(self, task: Task) -> ValidationReport:
        return await validate_task_live(
            task,
            sandbox_config=self.config.sandbox,
            expect_fail_on_starter=self.config.expect_fail_on_starter,
        )

    async def _generate_once(
        self,
        *,
        difficulty: str,
        domain: str,
        constraints: str,
    ) -> Task:
        if self._dspy_module is not None:
            return await self._generate_dspy(difficulty, domain, constraints)
        return await self._generate_openai_json(difficulty, domain, constraints)

    async def _generate_dspy(
        self, difficulty: str, domain: str, constraints: str
    ) -> Task:
        import asyncio

        pred = await asyncio.to_thread(
            self._dspy_module,
            difficulty=difficulty,
            domain=domain,
            constraints=constraints,
        )
        return parse_task_from_prediction(
            task_id=pred.task_id,
            description=pred.description,
            starter_files_json=pred.starter_files_json,
            setup_commands_json=pred.setup_commands_json,
            verifier_json=pred.verifier_json,
            metadata_json=pred.metadata_json,
            generator_model=self.config.model,
        )

    async def _generate_openai_json(
        self, difficulty: str, domain: str, constraints: str
    ) -> Task:
        client = build_openai_client(self.model_config)
        system = (
            "You generate self-contained coding tasks for agents in Docker. "
            "Reply with a single JSON object only (no markdown) with keys: "
            "task_id, description, starter_files (object path->content), "
            "setup_commands (list of strings), verifier (object with type/command/"
            "success_exit_code), metadata (difficulty, tags, estimated_steps, language). "
            "Starter code must FAIL the verifier; a correct fix should pass."
        )
        user = (
            f"difficulty={difficulty}\n"
            f"domain={domain}\n"
            f"constraints={constraints or 'none'}\n"
            "Prefer Python + pytest. Include complete file contents."
        )
        resp = await client.chat.completions.create(
            model=self.model_config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = _strip_code_fence(content)
        data = json.loads(content)
        return parse_task_from_prediction(
            task_id=data.get("task_id", f"gen_{uuid.uuid4().hex[:8]}"),
            description=data["description"],
            starter_files_json=json.dumps(data.get("starter_files") or {}),
            setup_commands_json=json.dumps(data.get("setup_commands") or []),
            verifier_json=json.dumps(data.get("verifier") or {"type": "pytest"}),
            metadata_json=json.dumps(data.get("metadata") or {"difficulty": difficulty}),
            generator_model=self.config.model,
        )


def _strip_code_fence(text: str) -> str:
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text.strip())
    if m:
        return m.group(1)
    return text
