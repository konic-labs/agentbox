"""Lenient JSON parsing for model-produced tool arguments.

Small models routinely emit raw newlines/tabs inside JSON strings (code!),
Python literals (True/None), single-quoted dicts or trailing commas.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_PY_LITERALS = ((re.compile(r"\bTrue\b"), "true"), (re.compile(r"\bFalse\b"), "false"),
                (re.compile(r"\bNone\b"), "null"))


def _escape_controls_in_strings(text: str) -> str:
    out: list[str] = []
    in_str = escape = False
    for ch in text:
        if escape:
            out.append(ch)
            escape = False
        elif ch == "\\":
            out.append(ch)
            escape = True
        elif ch == '"':
            in_str = not in_str
            out.append(ch)
        elif in_str and ch in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
        else:
            out.append(ch)
    return "".join(out)


def loads_lenient(text: str) -> Any:
    """``json.loads`` with fallbacks; raises ``ValueError`` if nothing works."""
    try:
        return json.loads(text, strict=False)  # allows raw control chars in strings
    except json.JSONDecodeError as first:
        candidates = [_escape_controls_in_strings(text)]
        repaired = _TRAILING_COMMA.sub(r"\1", text)
        for rx, repl in _PY_LITERALS:
            repaired = rx.sub(repl, repaired)
        candidates.append(repaired)
        for cand in candidates:
            try:
                return json.loads(cand, strict=False)
            except json.JSONDecodeError:
                continue
        try:
            return ast.literal_eval(text)  # single-quoted / Python dict syntax
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            pass
        raise ValueError(f"{first.msg} at char {first.pos}") from first
