"""Artifact store protocol + local filesystem backend."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    def put(self, src: Path, key: str) -> str: ...
    def get(self, key: str, dest: Path) -> Path: ...
    def exists(self, key: str) -> bool: ...


class LocalArtifactStore:
    """Store artifacts under a root directory: runs/, suites/, cache/."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # prevent path escape
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"invalid artifact key: {key}")
        return p

    def put(self, src: Path, key: str) -> str:
        src = Path(src)
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        return key

    def get(self, key: str, dest: Path) -> Path:
        src = self._path(key)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        return dest

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def run_dir(self, run_id: str) -> Path:
        p = self.root / "runs" / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def suite_dir(self, suite_id: str, content_hash: str) -> Path:
        p = self.root / "suites" / suite_id / content_hash
        p.mkdir(parents=True, exist_ok=True)
        return p
