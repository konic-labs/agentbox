"""DSPy signatures for task generation and LLM task validation."""

from __future__ import annotations

from typing import Any


def get_generate_coding_task_signature() -> Any:
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "dspy is required. Install with: pip install agentbox[generate]"
        ) from exc

    class GenerateCodingTask(dspy.Signature):
        """Author a coding task for an agent in a Docker sandbox.

        CRITICAL — starter must NOT be a near-complete solution:
        - Prefer stubs: signatures + docstrings + raise NotImplementedError
          (or empty/minimal bodies). The agent must WRITE the implementation.
        - Do NOT ship a full algorithm with one flipped operator as the "bug".
        - Do NOT put # BUG comments or "should be X instead of Y" in code.
        - description: required behavior + public API only (symptoms/tests).
          Forbidden: naming the exact operator, line, or one-line fix.

        Include complete starter file contents, setup commands
        (e.g. pip install -q pytest), and a deterministic pytest verifier.
        Starter must FAIL the verifier; a correct implementation must pass.
        Prefer compact files (under ~80 lines each).
        """

        difficulty: str = dspy.InputField(desc="easy | medium | hard")
        domain: str = dspy.InputField(desc="Task domain, e.g. python")
        constraints: str = dspy.InputField(desc="Extra constraints; empty if none")

        task_id: str = dspy.OutputField(desc="Unique slug id")
        description: str = dspy.OutputField(
            desc="Instruction shown to the agent (no solution spoilers)"
        )
        starter_files_json: str = dspy.OutputField(
            desc='JSON object path->content; stubs preferred, not near-solutions'
        )
        setup_commands_json: str = dspy.OutputField(
            desc='JSON list of shell commands, e.g. ["pip install -q pytest"]'
        )
        verifier_json: str = dspy.OutputField(
            desc='JSON VerifierSpec: {"type":"pytest","command":"python -m pytest -q",...}'
        )
        metadata_json: str = dspy.OutputField(
            desc="JSON metadata with difficulty, tags, estimated_steps, language"
        )

    return GenerateCodingTask


def get_validate_coding_task_signature() -> Any:
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "dspy is required. Install with: pip install agentbox[generate]"
        ) from exc

    class ValidateCodingTask(dspy.Signature):
        """Judge whether a coding task is suitable for agent training / benchmarks.

        ACCEPT only if:
        - starter_files are stubs or incomplete (agent must implement real logic)
        - description does not leak the exact fix (operator, line, one-token patch)
        - tests/verifier form a clear, solvable contract
        - task is non-trivial for a coding agent (not reverse/is_even one-liners only)

        REJECT if:
        - starter already contains a near-complete correct algorithm with a tiny bug
        - source or description contains # BUG / "should be" / exact operator spoilers
        - starter is empty garbage with no testable contract
        - schema junk paths (setup_commands, verifier, metadata as files)
        """

        task_id: str = dspy.InputField()
        description: str = dspy.InputField(desc="Agent-facing instruction")
        starter_files_json: str = dspy.InputField(
            desc="JSON object of path -> file content"
        )
        setup_commands_json: str = dspy.InputField(desc="JSON list of setup commands")
        verifier_json: str = dspy.InputField(desc="JSON verifier spec")
        difficulty: str = dspy.InputField(desc="Claimed difficulty")
        claimed_domain: str = dspy.InputField(desc="Domain e.g. python")

        accept: bool = dspy.OutputField(
            desc="True only if the task is a good agent coding task"
        )
        score: float = dspy.OutputField(
            desc=(
                "Quality score in buckets: 0.2=reject garbage, 0.4=weak/leaky, "
                "0.6=borderline stubs+tests, 0.8=solid agent task, 1.0=excellent. "
                "Avoid all-1.0; use the ladder."
            )
        )
        starter_is_near_solution: bool = dspy.OutputField(
            desc="True if non-test starter is nearly a full correct implementation"
        )
        description_leaks_fix: bool = dspy.OutputField(
            desc="True if description or comments spoil the exact fix"
        )
        agent_must_implement: bool = dspy.OutputField(
            desc="True if agent must write substantial logic (not one-token patch)"
        )
        reasons: str = dspy.OutputField(desc="Short reasons for accept/reject")
        suggested_fixes: str = dspy.OutputField(
            desc="How to improve the task if rejected; empty if accept"
        )

    return ValidateCodingTask


def get_generator_module() -> Any:
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "dspy is required. Install with: pip install agentbox[generate]"
        ) from exc

    Signature = get_generate_coding_task_signature()

    class CodingTaskGenerator(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.predict = dspy.Predict(Signature)

        def forward(self, difficulty: str, domain: str, constraints: str = "") -> Any:
            return self.predict(
                difficulty=difficulty,
                domain=domain,
                constraints=constraints or "",
            )

    return CodingTaskGenerator


def get_validator_module() -> Any:
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "dspy is required. Install with: pip install agentbox[generate]"
        ) from exc

    Signature = get_validate_coding_task_signature()

    class CodingTaskValidator(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.predict = dspy.Predict(Signature)

        def forward(
            self,
            task_id: str,
            description: str,
            starter_files_json: str,
            setup_commands_json: str,
            verifier_json: str,
            difficulty: str = "unknown",
            claimed_domain: str = "python",
        ) -> Any:
            return self.predict(
                task_id=task_id,
                description=description,
                starter_files_json=starter_files_json,
                setup_commands_json=setup_commands_json,
                verifier_json=verifier_json,
                difficulty=difficulty,
                claimed_domain=claimed_domain,
            )

    return CodingTaskValidator
