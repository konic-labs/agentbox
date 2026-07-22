"""Task definitions, seeding, and verification."""

from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.tasks.seeder import SeedResult, TaskSeeder
from agentbox.tasks.verifier import Verifier, VerifyResult

__all__ = [
    "Task",
    "VerifierSpec",
    "TaskSeeder",
    "SeedResult",
    "Verifier",
    "VerifyResult",
]
