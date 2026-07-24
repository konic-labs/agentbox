"""Automated task generation (optional dspy extra)."""

from agentbox.tasks.generate.batch import BatchGenerateConfig, batch_generate
from agentbox.tasks.generate.dedup import (
    difficulty_heuristic,
    is_near_duplicate,
    task_signature_hash,
)
from agentbox.tasks.generate.generator import GenerateConfig, TaskGenerator
from agentbox.tasks.generate.llm_validate import validate_task_llm
from agentbox.tasks.generate.review import (
    ReviewDecision,
    ReviewQueueItem,
    export_review_queue,
    filter_tasks_by_decisions,
    load_review_decisions,
)
from agentbox.tasks.generate.static_qc import StaticQCReport, validate_task_static
from agentbox.tasks.generate.strip_impl import strip_impl_files, strip_python_source
from agentbox.tasks.generate.validate import ValidationReport, validate_task_live

__all__ = [
    "BatchGenerateConfig",
    "GenerateConfig",
    "ReviewDecision",
    "ReviewQueueItem",
    "StaticQCReport",
    "TaskGenerator",
    "ValidationReport",
    "batch_generate",
    "difficulty_heuristic",
    "export_review_queue",
    "filter_tasks_by_decisions",
    "is_near_duplicate",
    "load_review_decisions",
    "strip_impl_files",
    "strip_python_source",
    "task_signature_hash",
    "validate_task_live",
    "validate_task_llm",
    "validate_task_static",
]
