"""Prompt materials for task generation domains."""

DOMAIN_HINTS: dict[str, str] = {
    "python": "Use Python 3.12, pytest for verification, keep files under /workspace.",
    "algorithms": "Classic algorithm bug-fix with clear unit tests.",
    "data": "Small pandas/numpy exercises with deterministic asserts.",
}
