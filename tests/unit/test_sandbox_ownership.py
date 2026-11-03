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


def test_cluster_prune_runs_once_on_every_alive_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """Orphans are judged on each node against its own processes: one pinned task per node."""
    import sys
    import types

    from agentbox.sandbox import prune as prune_mod

    calls: list[tuple[str, dict[str, Any]]] = []

    class Strategy:
        def __init__(self, node_id: str, soft: bool) -> None:
            assert soft is False
            self.node_id = node_id

    class Task:
        def __init__(self, fn: Any) -> None:
            self.fn, self.node = fn, ""

        def options(self, scheduling_strategy: Strategy) -> Task:
            self.node = scheduling_strategy.node_id
            return self

        def remote(self, kwargs: dict[str, Any]) -> Any:
            calls.append((self.node, kwargs))
            if self.node == "gpu-only":  # a node without Docker reports, it does not fail
                return {"host": self.node, "removed": [], "error": "DockerNotAvailableError: no"}
            return {"host": self.node, "removed": ["c1"], "error": None}

    ray = types.SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kw: None,
        shutdown=lambda: None,
        nodes=lambda: [
            {"NodeID": "head", "Alive": True},
            {"NodeID": "gpu-only", "Alive": True},
            {"NodeID": "gone", "Alive": False},
        ],
        remote=lambda **kw: Task,
        get=lambda refs: refs,
    )
    strategies = types.ModuleType("ray.util.scheduling_strategies")
    strategies.NodeAffinitySchedulingStrategy = Strategy  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", types.ModuleType("ray.util"))
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategies)
    out = prune_mod.prune_ray_cluster("auto", scope="orphans", dry_run=True)
    assert [c[0] for c in calls] == ["head", "gpu-only"]
    assert calls[0][1] == {"scope": "orphans", "dry_run": True}
    assert out[0]["removed"] == ["c1"] and out[1]["error"].startswith("DockerNotAvailable")
