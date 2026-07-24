"""Deterministically strip Python implementations to stubs (two-stage gen)."""

from __future__ import annotations

import ast
from typing import Mapping


class _StubTransformer(ast.NodeTransformer):
    """Replace function/method bodies with raise NotImplementedError."""

    def _stub_body(self, node: ast.AST) -> list[ast.stmt]:
        # preserve docstring if present
        body: list[ast.stmt] = []
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(node.body[0].value.value, str)  # type: ignore[union-attr]
        ):
            body.append(node.body[0])
        body.append(
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                ),
                cause=None,
            )
        )
        return body

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = self._stub_body(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.body = self._stub_body(node)
        return node


def strip_python_source(source: str) -> str:
    """Return source with function bodies replaced by NotImplementedError stubs."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # fallback: leave as-is; caller QC will catch
        return source
    new_tree = _StubTransformer().visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree) + "\n"
    except Exception:
        return source


def strip_impl_files(files: Mapping[str, str]) -> dict[str, str]:
    """Strip implementations for Python source files; leave tests unchanged."""
    out: dict[str, str] = {}
    for path, content in files.items():
        p = str(path)
        body = str(content)
        if p.endswith(".py") and "test" not in p.lower():
            out[p] = strip_python_source(body)
        else:
            out[p] = body
    return out
