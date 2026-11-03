"""Sandbox ownership across processes: owner labels and forked children (GH #98)."""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace
from typing import Any

import pytest

from agentbox.sandbox import backend as backend_mod


def test_owner_label_names_the_calling_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_mod.os, "getpid", lambda: 4242)
    assert backend_mod.owner_label() == f"{socket.gethostname()}:4242"
    be = backend_mod.DockerBackend(client=object())  # an injected client needs no daemon
    try:
        assert be._labels()["agentbox.owner"] == f"{socket.gethostname()}:4242"
        assert be.pid == 4242
    finally:
        be.close()


def test_labels_follow_a_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend created before fork labels the child's sandboxes with the child's pid."""
    be = backend_mod.DockerBackend(client=object())
    try:
        parent = be._labels()["agentbox.owner"]
        child_pid = os.getpid() + 1
        monkeypatch.setattr(backend_mod.os, "getpid", lambda: child_pid)
        child = be._labels()["agentbox.owner"]
        assert parent != child and child.endswith(f":{child_pid}")
    finally:
        monkeypatch.undo()
        be.close()


def test_forked_child_gets_its_own_shared_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    inherited = SimpleNamespace(closed=False, pid=os.getpid() + 1)
    made: list[Any] = []

    class Stub:
        def __init__(self) -> None:
            self.closed = False
            self.pid = os.getpid()
            made.append(self)

    monkeypatch.setattr(backend_mod, "_default_backend", inherited)
    monkeypatch.setattr(backend_mod, "_reaped", True)  # no background reaper thread
    monkeypatch.setattr(backend_mod, "DockerBackend", Stub)
    got = backend_mod.get_backend()
    assert got is made[0] and got is not inherited
    assert backend_mod.get_backend() is got  # cached for this process
    assert len(made) == 1
