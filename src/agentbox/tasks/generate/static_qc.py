"""Static quality checks for generated tasks (no LLM, no Docker)."""

from __future__ import annotations

import ast
import re
from typing import Any

from pydantic import BaseModel, Field

from agentbox.tasks.schema import Task

_JUNK_KEYS = {"setup_commands", "verifier", "metadata", "task_id", "description"}
_LEAK_PATTERNS = [
    re.compile(r"#\s*BUG\b", re.I),
    re.compile(r"\bshould be\b", re.I),
    re.compile(r"\binstead of\b", re.I),
    re.compile(r"`[><=!]+`\s*instead", re.I),
]


class StaticQCReport(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


def _non_test_files(task: Task) -> dict[str, str]:
    out: dict[str, str] = {}
    for path, content in (task.starter_files or {}).items():
        p = str(path)
        if p in _JUNK_KEYS or p.startswith("__"):
            continue
        if "test" in p.lower():
            continue
        out[p] = str(content)
    return out


def _test_files(task: Task) -> dict[str, str]:
    return {
        str(p): str(c)
        for p, c in (task.starter_files or {}).items()
        if "test" in str(p).lower()
    }


def _count_asserts(tests: dict[str, str]) -> int:
    n = 0
    for body in tests.values():
        n += len(re.findall(r"\bassert\b", body))
        n += len(re.findall(r"pytest\.raises", body))
    return n


def _has_stub_bodies(src: dict[str, str]) -> tuple[bool, list[str]]:
    """Return (ok, errors) — public functions should be stubs for coding tasks."""
    errors: list[str] = []
    found_def = False
    for path, body in src.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(body)
        except SyntaxError as exc:
            errors.append(f"{path}: syntax error in starter: {exc}")
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found_def = True
                if not _is_stub_function(node):
                    errors.append(
                        f"{path}:{node.name}: expected stub "
                        "(NotImplementedError / pass / ellipsis), found real body"
                    )
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        found_def = True
                        if item.name.startswith("_") and item.name != "__init__":
                            continue
                        if not _is_stub_function(item):
                            errors.append(
                                f"{path}:{node.name}.{item.name}: expected stub body"
                            )
    if not found_def and src:
        errors.append("no public function/class methods found in non-test starters")
    return (len(errors) == 0), errors


def _is_stub_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = [n for n in node.body if not isinstance(n, (ast.Expr, ast.Pass))]
    # docstring-only + pass/ellipsis/raise
    stmts = list(node.body)
    if stmts and isinstance(stmts[0], ast.Expr) and isinstance(
        getattr(stmts[0], "value", None), ast.Constant
    ):
        stmts = stmts[1:]
    if not stmts:
        return True
    if len(stmts) == 1:
        s = stmts[0]
        if isinstance(s, ast.Pass):
            return True
        if isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant):
            if s.value.value is ...:  # type: ignore[union-attr]
                return True
        if isinstance(s, ast.Raise):
            # raise NotImplementedError(...)
            return True
        if isinstance(s, ast.Return) and s.value is None:
            return True
    # allow docstring + raise NotImplementedError only
    if all(isinstance(s, (ast.Pass, ast.Raise)) for s in stmts):
        return True
    return False


def _path_consistency(task: Task) -> list[str]:
    errors: list[str] = []
    paths = set(task.starter_files.keys())
    # imports in tests referencing modules that aren't seeded
    for tpath, body in _test_files(task).items():
        for m in re.finditer(
            r"^\s*(?:from|import)\s+([a-zA-Z_][\w.]*)", body, re.M
        ):
            mod = m.group(1).split(".")[0]
            if mod in {"pytest", "unittest", "typing", "collections", "json", "os", "sys", "re", "math", "time", "pathlib", "dataclasses", "enum", "functools", "itertools", "copy", "random"}:
                continue
            # accept src.mod or mod.py
            candidates = {
                f"{mod}.py",
                f"src/{mod}.py",
                f"src/{mod}/__init__.py",
                mod.replace(".", "/") + ".py",
            }
            if not (candidates & paths) and not any(
                p.startswith(f"src/{mod}") or p.endswith(f"/{mod}.py") for p in paths
            ):
                # soft: only error if no path contains the module name
                if not any(mod in p for p in paths):
                    errors.append(
                        f"{tpath}: imports {mod} but no matching starter file among {sorted(paths)}"
                    )
    return errors


def _description_mentions_paths(task: Task) -> list[str]:
    """Warn/error when description never names any starter path."""
    files = list((task.starter_files or {}).keys())
    if not files:
        return []
    desc = task.description or ""
    if any(p in desc for p in files):
        return []
    # basename match
    basenames = [p.rsplit("/", 1)[-1] for p in files]
    if any(b in desc for b in basenames):
        return []
    return [
        "description does not mention any starter_files path "
        f"(expected one of {sorted(files)[:5]})"
    ]


def validate_task_static(
    task: Task,
    *,
    min_asserts: int = 3,
    require_stubs: bool = True,
    check_leaks: bool = True,
    max_file_lines: int = 200,
    strict_paths: bool = False,
) -> StaticQCReport:
    """Run static QC; returns ok=False if hard errors found."""
    errors: list[str] = []
    warnings: list[str] = []
    files = task.starter_files or {}

    junk = [p for p in files if p in _JUNK_KEYS]
    if junk:
        errors.append(f"junk starter keys: {junk}")

    src = _non_test_files(task)
    tests = _test_files(task)
    if not src:
        errors.append("no non-test starter source files")
    if not tests:
        warnings.append("no test files detected in starter_files")

    n_assert = _count_asserts(tests)
    if n_assert < min_asserts:
        errors.append(f"assert-like statements {n_assert} < min_asserts {min_asserts}")

    for path, body in files.items():
        lines = body.count("\n") + (1 if body else 0)
        if lines > max_file_lines:
            warnings.append(f"{path}: {lines} lines > max_file_lines {max_file_lines}")

    if require_stubs and src:
        ok_stub, stub_errs = _has_stub_bodies(src)
        if not ok_stub:
            errors.extend(stub_errs)

    if check_leaks:
        blob = task.description + "\n" + "\n".join(src.values())
        for pat in _LEAK_PATTERNS:
            if pat.search(blob):
                # description "instead of" is common English — only hard-fail on # BUG in source
                if pat.pattern.startswith(r"#"):
                    errors.append(f"leak pattern in source: {pat.pattern}")
                else:
                    warnings.append(f"possible leak pattern: {pat.pattern}")

    path_errs = _path_consistency(task)
    desc_errs = _description_mentions_paths(task)
    if strict_paths:
        errors.extend(path_errs)
        errors.extend(desc_errs)
    else:
        warnings.extend(path_errs)
        warnings.extend(desc_errs)

    return StaticQCReport(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        details={
            "n_asserts": n_assert,
            "n_src_files": len(src),
            "n_test_files": len(tests),
            "src_paths": sorted(src.keys()),
            "test_paths": sorted(tests.keys()),
        },
    )
