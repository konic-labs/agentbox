"""Near-duplicate detection for generated coding tasks."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Mapping

from agentbox.tasks.schema import Task

_IDENT = re.compile(r"\bdef\s+([a-zA-Z_][\w]*)\b")
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def task_signature_hash(task: Task) -> str:
    """Stable fingerprint of public APIs + test bodies (ignores stub bodies)."""
    files = task.starter_files or {}
    apis: list[str] = []
    tests: list[str] = []
    for path, body in sorted(files.items()):
        p = str(path)
        b = str(body)
        if "test" in p.lower():
            tests.append(_norm(b))
        elif p.endswith(".py"):
            apis.extend(_IDENT.findall(b))
    blob = "|".join(sorted(set(apis))) + "||" + "||".join(tests)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def minhash_tokens(task: Task, *, ngram: int = 3) -> set[str]:
    """Simple char-ngram shingles for Jaccard near-dup checks."""
    files = task.starter_files or {}
    text_parts: list[str] = []
    for path, body in sorted(files.items()):
        if "test" in str(path).lower() or str(path).endswith(".py"):
            text_parts.append(_norm(str(body)))
    text = " ".join(text_parts)
    if len(text) < ngram:
        return {text} if text else set()
    return {text[i : i + ngram] for i in range(len(text) - ngram + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def is_near_duplicate(
    task: Task,
    existing: Iterable[Task],
    *,
    jaccard_threshold: float = 0.85,
    exact_signature: bool = True,
) -> tuple[bool, str | None]:
    """Return (is_dup, reason)."""
    sig = task_signature_hash(task)
    tokens = minhash_tokens(task)
    for other in existing:
        if exact_signature and task_signature_hash(other) == sig:
            return True, f"exact signature match with {other.task_id}"
        score = jaccard(tokens, minhash_tokens(other))
        if score >= jaccard_threshold:
            return True, f"jaccard={score:.2f} with {other.task_id}"
    return False, None


def difficulty_heuristic(task: Task) -> dict[str, float | int | str]:
    """Cheap difficulty signals for batch filtering / reporting."""
    files = task.starter_files or {}
    n_assert = 0
    n_files = len(files)
    multi = n_files >= 3
    stateful = 0
    for path, body in files.items():
        b = str(body).lower()
        n_assert += len(re.findall(r"\bassert\b", b))
        n_assert += len(re.findall(r"pytest\.raises", b))
        for kw in ("state", "cache", "queue", "stack", "transition", "rate limit"):
            if kw in b:
                stateful += 1
    score = min(1.0, 0.15 * n_assert + (0.2 if multi else 0) + 0.05 * stateful)
    label = "easy" if score < 0.4 else ("medium" if score < 0.7 else "hard")
    return {
        "heuristic_score": round(score, 3),
        "heuristic_label": label,
        "n_asserts": n_assert,
        "n_files": n_files,
        "multi_file": int(multi),
        "stateful_hits": stateful,
    }


def load_existing_signatures(tasks: Iterable[Task]) -> dict[str, str]:
    """Map signature hash -> task_id for resume sets."""
    out: dict[str, str] = {}
    for t in tasks:
        out[task_signature_hash(t)] = t.task_id
    return out
