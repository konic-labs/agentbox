"""DSPy signatures for task generation (optional dependency)."""

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
        """Generate a self-contained coding task for an agent in a Docker sandbox.

        The task must include complete starter file contents, optional setup commands
        (e.g. pip install), a clear agent instruction, and a deterministic verifier
        (preferably pytest). The initial environment should FAIL the verifier;
        a correct solution should make the verifier pass.
        """

        difficulty: str = dspy.InputField(desc="easy | medium | hard")
        domain: str = dspy.InputField(desc="Task domain, e.g. python")
        constraints: str = dspy.InputField(desc="Extra constraints; empty if none")

        task_id: str = dspy.OutputField(desc="Unique slug id")
        description: str = dspy.OutputField(desc="Instruction shown to the agent")
        starter_files_json: str = dspy.OutputField(
            desc='JSON object mapping path to full file content, e.g. {"main.py":"..."}'
        )
        setup_commands_json: str = dspy.OutputField(
            desc='JSON list of shell commands, e.g. ["pip install -q pytest"]'
        )
        verifier_json: str = dspy.OutputField(
            desc='JSON VerifierSpec: {"type":"pytest","command":"python -m pytest -q",...}'
        )
        metadata_json: str = dspy.OutputField(
            desc='JSON metadata with difficulty, tags, estimated_steps, language'
        )

    return GenerateCodingTask


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
