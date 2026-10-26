"""apply_patch: apply a unified diff on the host side.

Slim images ship neither ``patch`` nor ``git``, so the diff is parsed and
applied in Python and the results are written back through the sandbox. All
files are validated first; nothing is written unless every hunk applies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agentbox.tools.base import BaseTool, ToolResult

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(ValueError):
    pass


@dataclass
class Hunk:
    old_start: int
    lines: list[str] = field(default_factory=list)  # " x", "-x", "+x"

    @property
    def old_lines(self) -> list[str]:
        return [ln[1:] for ln in self.lines if ln[:1] in (" ", "-")]

    @property
    def new_lines(self) -> list[str]:
        return [ln[1:] for ln in self.lines if ln[:1] in (" ", "+")]


@dataclass
class FilePatch:
    old_path: str | None
    new_path: str | None
    hunks: list[Hunk] = field(default_factory=list)


def _clean_path(raw: str) -> str | None:
    p = raw.strip().split("\t", 1)[0].strip()
    if p == "/dev/null":
        return None
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


def parse_unified_diff(text: str) -> list[FilePatch]:
    patches: list[FilePatch] = []
    current: FilePatch | None = None
    hunk: Hunk | None = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            current = FilePatch(_clean_path(line[4:]), _clean_path(lines[i + 1][4:]))
            patches.append(current)
            hunk = None
            i += 2
            continue
        m = _HUNK_RE.match(line)
        if m:
            if current is None:
                raise PatchError("hunk before file header (---/+++)")
            hunk = Hunk(old_start=int(m.group(1)))
            current.hunks.append(hunk)
            i += 1
            continue
        if hunk is not None and line[:1] in (" ", "-", "+"):
            hunk.lines.append(line)
        elif hunk is not None and line == "":
            hunk.lines.append(" ")  # blank context line whose leading space was stripped
        # "\ No newline at end of file", "diff --git", "index ..." are ignored
        i += 1
    if not patches:
        raise PatchError("no file headers found (expected '--- a/path' / '+++ b/path')")
    return patches


def _find(haystack: list[str], needle: list[str], hint: int) -> int:
    if not needle:
        return max(0, min(hint, len(haystack)))
    n = len(needle)
    if haystack[hint : hint + n] == needle:
        return hint
    hits = [i for i in range(len(haystack) - n + 1) if haystack[i : i + n] == needle]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return -1
    return min(hits, key=lambda i: abs(i - hint))  # ambiguous: closest to header


def apply_hunks(original: str, hunks: list[Hunk]) -> str:
    had_newline = original.endswith("\n") or original == ""
    lines = original.splitlines()
    offset = 0
    for h in hunks:
        old, new = h.old_lines, h.new_lines
        pos = _find(lines, old, max(0, h.old_start - 1 + offset))
        if pos < 0:
            preview = "\n".join(old[:3])
            raise PatchError(f"hunk @@ -{h.old_start} does not apply; expected:\n{preview}")
        lines[pos : pos + len(old)] = new
        offset += len(new) - len(old)
    out = "\n".join(lines)
    return out + "\n" if had_newline and lines else out


class ApplyPatchTool(BaseTool):
    name = "apply_patch"
    description = (
        "Apply a unified diff (as produced by `diff -u` / `git diff`) to workspace files. "
        "Supports creating (--- /dev/null) and deleting (+++ /dev/null) files. "
        "Either every hunk applies or nothing is changed."
    )

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"patch": {"type": "string", "description": "Unified diff text"}},
            "required": ["patch"],
            "additionalProperties": False,
        }

    async def execute(self, sandbox: Any, patch: str, **_: Any) -> str | ToolResult:
        try:
            file_patches = parse_unified_diff(patch)
        except PatchError as exc:
            return ToolResult(content=f"ERROR: apply_patch failed: {exc}", is_error=True)
        writes: dict[str, str] = {}
        deletes: list[str] = []
        try:
            for fp in file_patches:
                if fp.new_path is None:
                    if fp.old_path is None:
                        raise PatchError("both paths are /dev/null")
                    deletes.append(fp.old_path)
                    continue
                if fp.old_path is None:
                    original = ""
                else:
                    try:
                        original = await sandbox.read_text(fp.old_path)
                    except FileNotFoundError as exc:
                        raise PatchError(f"file not found: {fp.old_path}") from exc
                writes[fp.new_path] = apply_hunks(original, fp.hunks)
                if fp.old_path and fp.old_path != fp.new_path:
                    deletes.append(fp.old_path)
        except PatchError as exc:
            return ToolResult(content=f"ERROR: apply_patch failed: {exc}", is_error=True)
        except Exception as exc:
            return ToolResult(content=f"ERROR: apply_patch failed: {exc}", is_error=True)
        if writes:
            await sandbox.write_files(writes)
        if deletes:
            res = await sandbox.exec(["rm", "-f", "--", *deletes], timeout_s=30.0)
            if res.exit_code != 0:
                return ToolResult(
                    content=f"ERROR: apply_patch failed deleting files: {res.stderr.strip()}",
                    is_error=True,
                )
        summary = [f"M {p}" for p in writes] + [f"D {p}" for p in deletes]
        return "Applied patch:\n" + "\n".join(summary)
