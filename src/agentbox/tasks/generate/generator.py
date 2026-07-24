"""Automated task generation via teacher models (+ optional DSPy).

Supports self-hosted OpenAI-compatible endpoints (vLLM / TGI / etc.) for both
generation and LLM task validation (same model by default).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentbox.config import ModelConfig, SandboxConfig
from agentbox.errors import TaskGenerationError
from agentbox.tasks.generate.llm_validate import (
    merge_validation_reports,
    validate_task_llm,
)
from agentbox.tasks.generate.lm import build_dspy_lm, build_openai_client, configure_dspy_lm
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
    # Second API call: DSPy structured judge (same teacher by default)
    validate_with_llm: bool = True
    llm_judge_min_score: float = 0.65
    # Optional override for judge model (defaults to generation model/endpoint)
    validator_model: str | None = None
    validator_base_url: str | None = None
    validator_api_key: str | None = None
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
    # Extra OpenAI-compatible body fields
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # Auto-disable deep-thinking for structured gen/judge when possible
    disable_thinking: bool = True
    # Static QC before docker/llm (stubs, asserts, leaks)
    validate_static: bool = True
    static_min_asserts: int = 3
    static_require_stubs: bool = True
    static_strict_paths: bool = False
    # Two-stage: teacher emits solution_files; we AST-strip to starter stubs
    two_stage: bool = False
    # Keep golden solution in metadata (not in suite starter by default)
    keep_solution_in_metadata: bool = False
    # Optional content-addressed QC cache root
    validation_cache_dir: str | None = None


def _default_thinking_extra(model: str, disable_thinking: bool) -> dict[str, Any]:
    """Disable deep-thinking for structured generation (faster, non-empty text)."""
    if not disable_thinking:
        return {}
    mid = model.lower()
    if "glm" in mid or "zai-org" in mid:
        return {
            "thinking": {"type": "disabled"},
            "enable_thinking": False,
        }
    if "qwen" in mid:
        return {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    return {}


class TaskGenerator:
    """Generate complete Task definitions using a teacher model."""

    def __init__(self, config: GenerateConfig) -> None:
        self.config = config
        extra = dict(config.extra_body or {})
        for k, v in _default_thinking_extra(config.model, config.disable_thinking).items():
            extra.setdefault(k, v)
        self.model_config = ModelConfig(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            extra_body=extra,
        )
        # Judge uses same endpoint/model unless explicitly overridden
        v_model = config.validator_model or config.model
        v_extra = dict(config.extra_body or {})
        for k, v in _default_thinking_extra(v_model, config.disable_thinking).items():
            v_extra.setdefault(k, v)
        self.validator_model_config = ModelConfig(
            model=v_model,
            base_url=config.validator_base_url or config.base_url,
            api_key=config.validator_api_key or config.api_key,
            temperature=min(config.temperature, 0.3),
            # Same completion budget as generator (e.g. 8192 on this vLLM box)
            max_tokens=config.max_tokens or 8192,
            timeout_s=config.timeout_s,
            extra_body=v_extra,
        )
        self._dspy_module: Any | None = None
        self._dspy_lm: Any | None = None
        self._dspy_validator_lm: Any | None = None
        if config.use_dspy:
            try:
                from agentbox.tasks.generate.signatures import get_generator_module

                # Build LMs once; concurrent calls use dspy.context(lm=...) — not re-configure
                self._dspy_lm = build_dspy_lm(self.model_config)
                self._dspy_validator_lm = build_dspy_lm(self.validator_model_config)
                try:
                    configure_dspy_lm(self.model_config, set_global=True)
                except Exception:
                    pass
                self._dspy_module = get_generator_module()()
            except Exception as exc:
                logger.warning("DSPy unavailable (%s); falling back to OpenAI JSON", exc)
                self._dspy_module = None
                self._dspy_lm = None
                self._dspy_validator_lm = None

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

                if self.config.validate_static:
                    from agentbox.tasks.generate.static_qc import validate_task_static

                    static = validate_task_static(
                        task,
                        min_asserts=self.config.static_min_asserts,
                        require_stubs=self.config.static_require_stubs,
                        strict_paths=self.config.static_strict_paths,
                    )
                    if not static.ok:
                        last_error = "; ".join(static.errors) or "static QC failed"
                        logger.warning(
                            "generate attempt %d static QC: %s", attempt + 1, last_error
                        )
                        continue

                report = await self.validate_task(task, domain=domain)
                if not report.ok:
                    last_error = "; ".join(report.errors) or "validation failed"
                    logger.warning(
                        "generate attempt %d failed QC: %s", attempt + 1, last_error
                    )
                    continue
                # attach judge metadata for provenance
                if report.llm_score is not None:
                    task.metadata["llm_judge_score"] = report.llm_score
                    task.metadata["llm_judge_accept"] = report.llm_accept
                from agentbox.tasks.generate.dedup import (
                    difficulty_heuristic,
                    task_signature_hash,
                )

                task.metadata["signature_hash"] = task_signature_hash(task)
                task.metadata.update(
                    {k: v for k, v in difficulty_heuristic(task).items()}
                )
                # Drop golden solution from shipped task unless explicitly kept.
                # Only strip after Docker QC consumed it; batch mode disables Docker
                # here and needs solution_files for its own golden-pass stage.
                if (
                    task.metadata.get("solution_files")
                    and not self.config.keep_solution_in_metadata
                    and self.config.validate_in_docker
                ):
                    task.metadata.pop("solution_files", None)
                    task.metadata.pop("_solution_ephemeral", None)
                    # keep has_hidden_solution flag as provenance only
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

    async def validate_task(
        self,
        task: Task,
        *,
        domain: str = "python",
    ) -> ValidationReport:
        """Run configured QC stages: Docker fail-on-starter and/or LLM judge."""
        cache = None
        if self.config.validation_cache_dir:
            from pathlib import Path

            from agentbox.jobs.cache import ValidationCache

            cache = ValidationCache(Path(self.config.validation_cache_dir))
            hit = cache.get(task)
            if hit is not None:
                try:
                    return ValidationReport.model_validate(hit)
                except Exception:
                    pass

        reports: list[ValidationReport] = []
        solution_files = None
        if task.metadata.get("has_hidden_solution") and task.metadata.get(
            "solution_files"
        ):
            solution_files = {
                str(k): str(v) for k, v in task.metadata["solution_files"].items()
            }

        if self.config.validate_in_docker:
            docker_report = await validate_task_live(
                task,
                sandbox_config=self.config.sandbox,
                expect_fail_on_starter=self.config.expect_fail_on_starter,
                solution_files=solution_files,
            )
            reports.append(docker_report)
            if not docker_report.ok:
                return docker_report

        if self.config.validate_with_llm:
            llm_report = await validate_task_llm(
                task,
                model_config=self.validator_model_config,
                min_score=self.config.llm_judge_min_score,
                use_dspy=self.config.use_dspy,
                domain=domain,
                dspy_lm=self._dspy_validator_lm or self._dspy_lm,
            )
            reports.append(llm_report)

        if not reports:
            result = ValidationReport(ok=True, task=task, warnings=["no QC stages enabled"])
        elif len(reports) == 1:
            result = reports[0]
        else:
            result = merge_validation_reports(*reports, task=task)

        if cache is not None:
            try:
                cache.put(task, result.model_dump(mode="json"))
            except Exception:
                pass
        return result

    async def validate_task_llm_only(
        self,
        task: Task,
        *,
        domain: str = "python",
    ) -> ValidationReport:
        """LLM judge only (skip Docker). Useful for batch re-audit of existing tasks."""
        return await validate_task_llm(
            task,
            model_config=self.validator_model_config,
            min_score=self.config.llm_judge_min_score,
            use_dspy=self.config.use_dspy,
            domain=domain,
            dspy_lm=self._dspy_validator_lm or self._dspy_lm,
        )

    async def _generate_once(
        self,
        *,
        difficulty: str,
        domain: str,
        constraints: str,
    ) -> Task:
        if self.config.two_stage:
            task = await self._generate_two_stage(difficulty, domain, constraints)
        elif self._dspy_module is not None:
            task = await self._generate_dspy(difficulty, domain, constraints)
        else:
            task = await self._generate_openai_json(difficulty, domain, constraints)
        return task

    async def _generate_two_stage(
        self, difficulty: str, domain: str, constraints: str
    ) -> Task:
        """Generate golden solution+tests, then AST-strip solution → stubs."""
        from agentbox.tasks.generate.strip_impl import strip_impl_files

        # Prefer OpenAI JSON for structured solution_files; more reliable than DSPy field
        task = await self._generate_openai_json(
            difficulty,
            domain,
            (constraints or "")
            + " TWO_STAGE=1: include solution_files (full correct impl) AND "
            "starter will be derived by stripping. Put full impl under solution_files "
            "key in JSON; starter_files may equal solution or be omitted for non-tests.",
            two_stage=True,
        )
        solution = task.metadata.get("solution_files")
        if not isinstance(solution, dict) or not solution:
            # Fall back: treat non-test starter as solution and strip
            solution = {
                p: c
                for p, c in (task.starter_files or {}).items()
                if "test" not in str(p).lower()
            }
            tests = {
                p: c
                for p, c in (task.starter_files or {}).items()
                if "test" in str(p).lower()
            }
            if not solution:
                raise ValueError("two-stage generation produced no solution_files")
            stripped = strip_impl_files(solution)
            task.starter_files = {**stripped, **tests}
        else:
            solution = {str(k): str(v) for k, v in solution.items()}
            tests = {
                p: c
                for p, c in (task.starter_files or {}).items()
                if "test" in str(p).lower()
            }
            # Keep test files from starter; strip solution → starter sources
            stripped = strip_impl_files(solution)
            # If teacher also put tests only in solution_files, pull them out
            sol_tests = {
                p: c for p, c in solution.items() if "test" in str(p).lower()
            }
            sol_src = {
                p: c for p, c in solution.items() if "test" not in str(p).lower()
            }
            if sol_tests and not tests:
                tests = sol_tests
            if sol_src:
                stripped = strip_impl_files(sol_src)
                solution = sol_src
            task.starter_files = {**stripped, **tests}

        task.metadata["has_hidden_solution"] = True
        task.metadata["two_stage"] = True
        if self.config.keep_solution_in_metadata:
            task.metadata["solution_files"] = solution
        else:
            # Keep for Docker golden-pass during this generate() call only;
            # strip before save if user re-saves later without flag.
            task.metadata["solution_files"] = solution
            task.metadata["_solution_ephemeral"] = True
        return task

    async def _generate_dspy(
        self, difficulty: str, domain: str, constraints: str
    ) -> Task:
        import asyncio

        import dspy

        module = self._dspy_module
        lm = self._dspy_lm

        def _run() -> Any:
            if lm is not None:
                with dspy.context(lm=lm):
                    return module(
                        difficulty=difficulty,
                        domain=domain,
                        constraints=constraints,
                    )
            return module(
                difficulty=difficulty,
                domain=domain,
                constraints=constraints,
            )

        pred = await asyncio.to_thread(_run)
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
        self,
        difficulty: str,
        domain: str,
        constraints: str,
        *,
        two_stage: bool = False,
    ) -> Task:
        client = build_openai_client(self.model_config)
        if two_stage:
            system = (
                "You generate self-contained coding tasks for agents in Docker. "
                "Reply with ONE valid JSON object only (no markdown fences, no prose). "
                "Keys: task_id (string), description (string), "
                "solution_files (object path->FULL correct Python implementation), "
                "starter_files (object path->content; include ALL test_*.py files here; "
                "non-test sources may mirror solution or be omitted), "
                "setup_commands (array of strings), "
                "verifier (object with type/command/success_exit_code), "
                "metadata (object with difficulty, tags, estimated_steps, language). "
                "JSON rules: escape newlines as \\n, escape quotes as \\\", no trailing commas. "
                "Description states API/behavior only — no spoilers. "
                "solution_files must make pytest pass; tests must be thorough (>=3 asserts). "
                "Prefer Python + pytest. Name exact file paths consistently."
            )
            user = (
                f"difficulty={difficulty}\n"
                f"domain={domain}\n"
                f"constraints={constraints or 'none'}\n"
                "Return valid JSON only. Include solution_files (full impl) + test files."
            )
        else:
            system = (
                "You generate self-contained coding tasks for agents in Docker. "
                "Reply with ONE valid JSON object only (no markdown fences, no prose). "
                "Keys: task_id (string), description (string), "
                "starter_files (object mapping path string -> file content string), "
                "setup_commands (array of strings), "
                "verifier (object with type/command/success_exit_code), "
                "metadata (object with difficulty, tags, estimated_steps, language). "
                "JSON rules: escape all newlines in strings as \\n, escape quotes as \\\", "
                "no trailing commas, no raw control characters inside strings. "
                "CRITICAL: starter_files must be stubs (signatures + raise NotImplementedError). "
                "Do NOT ship near-complete solutions or # BUG spoilers. "
                "Description states API/behavior only. Prefer Python + pytest. "
                "Starter must FAIL tests; a correct implementation must pass."
            )
            user = (
                f"difficulty={difficulty}\n"
                f"domain={domain}\n"
                f"constraints={constraints or 'none'}\n"
                "Return valid JSON only. Starter = stubs, not almost-solutions."
            )
        create_kwargs: dict[str, Any] = {
            "model": self.model_config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.model_config.temperature,
            "max_tokens": self.model_config.max_tokens,
            # Helps vLLM / OpenAI-compat servers emit parseable JSON
            "response_format": {"type": "json_object"},
        }
        if self.model_config.extra_body:
            create_kwargs["extra_body"] = self.model_config.extra_body
        try:
            resp = await client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            # Some servers reject response_format — retry without it
            if "response_format" in str(exc).lower() or "json_object" in str(exc).lower():
                create_kwargs.pop("response_format", None)
                resp = await client.chat.completions.create(**create_kwargs)
            else:
                raise
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        if not content:
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            if isinstance(reasoning, str) and reasoning.strip():
                content = reasoning.strip()
        data = _parse_json_object(content)
        # Normalize nested objects → JSON strings for shared parser
        starter = data.get("starter_files") or {}
        solution = data.get("solution_files") or {}
        if isinstance(starter, str):
            starter_json = starter
        else:
            # If two-stage and starter empty, seed with solution so parser has files
            if two_stage and not starter and isinstance(solution, dict) and solution:
                starter = dict(solution)
            starter_json = json.dumps(starter)
        setup = data.get("setup_commands") or []
        setup_json = setup if isinstance(setup, str) else json.dumps(setup)
        verifier = data.get("verifier") or {"type": "pytest"}
        verifier_json = verifier if isinstance(verifier, str) else json.dumps(verifier)
        meta = data.get("metadata") or {"difficulty": difficulty}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {"difficulty": difficulty}
        if not isinstance(meta, dict):
            meta = {"difficulty": difficulty}
        if two_stage and isinstance(solution, dict) and solution:
            meta["solution_files"] = {str(k): str(v) for k, v in solution.items()}
            meta["has_hidden_solution"] = True
        # Path layout hints for agents / static QC
        if isinstance(starter, dict) and starter:
            paths = sorted(str(p) for p in starter.keys())
            meta.setdefault("layout", "src" if any(p.startswith("src/") for p in paths) else "flat")
            non_test = [p for p in paths if "test" not in p.lower() and p.endswith(".py")]
            if non_test:
                meta.setdefault("primary_module", non_test[0])
            meta.setdefault(
                "test_paths",
                [p for p in paths if "test" in p.lower()],
            )
        meta_json = json.dumps(meta)
        return parse_task_from_prediction(
            task_id=data.get("task_id", f"gen_{uuid.uuid4().hex[:8]}"),
            description=data["description"],
            starter_files_json=starter_json,
            setup_commands_json=setup_json,
            verifier_json=verifier_json,
            metadata_json=meta_json,
            generator_model=self.config.model,
        )


def _strip_code_fence(text: str) -> str:
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text.strip())
    if m:
        return m.group(1)
    return text


def _repair_json_text(text: str) -> str:
    """Best-effort fixes for common LLM JSON mistakes."""
    # trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # python-ish True/False/None
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model text, including CoT-wrapped / slightly broken JSON."""
    text = _strip_code_fence((text or "").strip())
    if not text:
        raise ValueError("empty model response")

    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    last_err: Exception | None = None
    for cand in candidates:
        for variant in (cand, _repair_json_text(cand)):
            try:
                data = json.loads(variant, strict=False)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError as exc:
                last_err = exc
                continue

    # Last resort: escape raw control chars inside strings (very common with code blobs)
    try:
        repaired = _escape_raw_controls_in_strings(candidates[-1])
        data = json.loads(_repair_json_text(repaired), strict=False)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        last_err = exc

    # Optional json_repair dependency
    try:
        from json_repair import repair_json  # type: ignore

        data = repair_json(candidates[-1], return_objects=True)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    raise ValueError(
        f"could not parse JSON object from model response ({len(text)} chars): {last_err}"
    )


def _escape_raw_controls_in_strings(text: str) -> str:
    """Escape raw newlines/tabs that appear inside JSON string literals."""
    out: list[str] = []
    in_str = False
    escape = False
    for ch in text:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            continue
        if in_str and ch == "\n":
            out.append("\\n")
            continue
        if in_str and ch == "\r":
            out.append("\\r")
            continue
        if in_str and ch == "\t":
            out.append("\\t")
            continue
        out.append(ch)
    return "".join(out)
