"""Sandboxes that die with the process that created them, with a fake Docker API (GH #111)."""

from __future__ import annotations

import os
import socket
import stat
from typing import Any

import pytest
from docker.errors import APIError

from agentbox.config import SandboxConfig
from agentbox.errors import SandboxError
from agentbox.sandbox import backend as backend_mod
from agentbox.sandbox.backend import ContainerSpec, DockerBackend, _Lease
from agentbox.sandbox.manager import SandboxManager


class _Raw:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False

    def sendall(self, data: bytes) -> None:
        assert not self.closed, "wrote to a closed stream"
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True


class _SocketIO:
    """Like the read-only SocketIO the unix transport returns."""

    def __init__(self) -> None:
        self._sock = _Raw()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Response:
    closed = False

    def close(self) -> None:
        self.closed = True


class _Api:
    def __init__(self, *, attach_error: bool = False) -> None:
        self.calls: list[str] = []
        self.create_kwargs: dict[str, Any] = {}
        self.host_kwargs: dict[str, Any] = {}
        self.responses: list[_Response] = []
        self.socks: list[_SocketIO] = []
        self.renamed: dict[str, str] = {}
        self.attach_error = attach_error

    def create_host_config(self, **kw: Any) -> dict[str, Any]:
        self.host_kwargs = kw
        return kw

    def create_container(self, **kw: Any) -> dict[str, str]:
        self.calls.append("create")
        self.create_kwargs = kw
        return {"Id": "c1"}

    def _url(self, path: str, *args: str) -> str:
        return path.format(*args)

    def post(self, url: str, data: Any, **kw: Any) -> _Response:
        self.calls.append(f"attach {url} {kw['params']} {kw['headers']['Upgrade']}")
        resp = _Response()
        self.responses.append(resp)
        return resp

    def _get_raw_response_socket(self, resp: _Response) -> _SocketIO:
        if self.attach_error:
            raise APIError("attach refused")
        sock = _SocketIO()
        self.socks.append(sock)
        return sock

    def start(self, cid: str) -> None:
        self.calls.append("start")

    def remove_container(self, cid: str, **kw: Any) -> None:
        self.calls.append("remove")

    def rename(self, cid: str, name: str) -> None:
        self.renamed[cid] = name


class _Client:
    def __init__(self, api: _Api) -> None:
        self.api = api


def _backend(api: _Api) -> DockerBackend:
    return DockerBackend(client=_Client(api))


def _spec(bound: bool = True) -> ContainerSpec:
    return ContainerSpec(
        image="img",
        name="agentbox-t-1",
        command=["sleep", "infinity"],
        working_dir="/workspace",
        labels={"agentbox": "1"},
        die_with_owner=bound,
    )


def test_a_bound_sandbox_is_attached_before_it_starts() -> None:
    api = _Api()
    be = _backend(api)
    try:
        assert be.create_container_sync(_spec()) == "c1"
        assert api.calls == [
            "create",
            "attach /containers/c1/attach {'stdin': 1, 'stream': 1} tcp",
            "start",
        ]
        kw = api.create_kwargs
        assert kw["stdin_open"] is True and kw["detach"] is False  # docker-py: StdinOnce
        assert kw["command"][:2] == ["/bin/sh", "-c"] and kw["command"][-2:] == [
            "sleep",
            "infinity",
        ]
        assert api.host_kwargs["auto_remove"] is True
        assert be.leased_containers() == ["c1"]
    finally:
        be.close()


def test_removal_releases_the_lease() -> None:
    api = _Api()
    be = _backend(api)
    be.create_container_sync(_spec())
    be.remove_container_sync("c1")
    assert api.calls[-1] == "remove"
    assert api.responses[0].closed and api.socks[0].closed
    assert be.leased_containers() == [] and be.live_containers() == []
    be.close()


def test_keep_tells_the_container_to_stay_before_letting_go() -> None:
    api = _Api()
    be = _backend(api)
    be.create_container_sync(_spec())
    assert be.keep_container_sync("c1", "agentbox-t-1") == "agentbox-t-1-kept"
    raw = api.socks[0]._sock
    assert raw.sent == [b"keep\n"]  # written while the stream was still open
    assert api.responses[0].closed and api.socks[0].closed
    assert be.leased_containers() == [] and be.live_containers() == []
    be.close()
    assert "remove" not in api.calls  # exit cleanup leaves a kept sandbox alone


def test_a_failed_attach_removes_the_container() -> None:
    api = _Api(attach_error=True)
    be = _backend(api)
    with pytest.raises(SandboxError, match="attach refused"):
        be.create_container_sync(_spec())
    assert "start" not in api.calls and api.calls[-1] == "remove"
    assert api.responses[0].closed and be.leased_containers() == []
    be.close()


def test_unbound_sandboxes_are_created_as_before() -> None:
    api = _Api()
    be = _backend(api)
    be.create_container_sync(_spec(bound=False))
    kw = api.create_kwargs
    assert kw["detach"] is True and kw["stdin_open"] is False
    assert kw["command"] == ["sleep", "infinity"] and "auto_remove" not in api.host_kwargs
    assert api.calls == ["create", "start"] and be.leased_containers() == []
    be.close()


def test_sandbox_config_binds_by_default() -> None:
    api = _Api()
    be = _backend(api)
    try:
        assert (
            SandboxManager(SandboxConfig(), backend=be)
            .container_spec(image="img", name="n", labels={})
            .die_with_owner
        )
        assert (
            not SandboxManager(SandboxConfig(die_with_owner=False), backend=be)
            .container_spec(image="img", name="n", labels={})
            .die_with_owner
        )
    finally:
        be.close()


def test_a_forked_child_lets_go_of_its_parents_streams() -> None:
    """After fork, the child swaps /dev/null over each lease: the parent alone holds it."""
    ours, theirs = socket.socketpair()
    be = _backend(_Api())
    try:
        be._leases["c9"] = _Lease(_Response(), ours)
        backend_mod._release_inherited_leases()  # what os.fork runs in the child
        assert stat.S_ISCHR(os.fstat(ours.fileno()).st_mode)  # now /dev/null
        theirs.settimeout(2)
        assert theirs.recv(1) == b""  # the stream's last holder let go: EOF
    finally:
        be._leases.clear()
        be.close()
        ours.close()
        theirs.close()
