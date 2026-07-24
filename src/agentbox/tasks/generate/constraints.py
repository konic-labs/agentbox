"""Default constraint families for batch task generation."""

from __future__ import annotations

from pathlib import Path

DEFAULT_FAMILIES = [
    "family=text-processing; implement a small parser/normalizer from pytest",
    "family=data-structures; implement mini stack/queue/set API from pytest",
    "family=algorithms; implement binary search / sliding window / group-by from pytest",
    "family=validation; multi-rule validator returning structured errors from pytest",
    "family=aggregation; group/count/sum-by-key over records from pytest",
    "family=path-utils; path join/normalize helpers with edge cases from pytest",
    "family=multi-file; src/module.py + tests/; implement 2+ related functions",
    "family=state-machine; small state transition or rate-limit logic from pytest",
]

SIZE_RULES = (
    "STARTER MUST BE STUBS ONLY: signatures + docstrings + raise NotImplementedError. "
    "Do NOT ship near-complete solutions or # BUG comments. "
    "Description states API/behavior only — no operator spoilers. "
    "Name exact file paths used in starter_files. "
    "Full pytest contract (>=4 asserts, edge cases). Starter fails; correct impl passes. "
    "NOT trivial one-liners (no bare add/multiply/is_even). stdlib + pytest only."
)


def constraint_for_index(i: int, *, families: list[str] | None = None) -> tuple[str, str]:
    fams = families or DEFAULT_FAMILIES
    difficulty = "easy" if i % 3 != 2 else "medium"
    family = fams[i % len(fams)]
    constraints = f"{family} {SIZE_RULES} unique_id_hint=task{i:02d}"
    return difficulty, constraints


def load_constraint_families(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_FAMILIES)
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    return lines or list(DEFAULT_FAMILIES)
