"""Text truncation that keeps both ends (errors and summaries live at the tail)."""

from __future__ import annotations


def truncate_middle(text: str, limit: int, *, label: str = "chars") -> str:
    """Keep the first and last ``limit // 2`` characters with an explicit marker.

    Tool output and test logs put the useful part (tracebacks, pytest summaries)
    at the end, so head-only truncation hides exactly what an agent needs.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    dropped = len(text) - head - tail
    return f"{text[:head]}\n...[truncated {dropped} {label}]...\n{text[-tail:]}"


def truncate_middle_bytes(data: bytes, limit: int) -> tuple[bytes, bool]:
    """Byte variant of :func:`truncate_middle`; returns ``(data, truncated)``."""
    if limit <= 0 or len(data) <= limit:
        return data, False
    head = limit // 2
    tail = limit - head
    dropped = len(data) - head - tail
    marker = f"\n...[truncated {dropped} bytes]...\n".encode()
    return data[:head] + marker + data[-tail:], True
