"""scripts/check_docs.py: links, anchors and images in the README and docs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_docs", Path(__file__).resolve().parents[2] / "scripts" / "check_docs.py"
)
assert _SPEC and _SPEC.loader
check_docs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_docs)


def write(root: Path, rel: str, text: str | bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def site(tmp_path: Path) -> Path:
    write(tmp_path, "docs/assets/readme/banner.svg", "<svg/>")
    write(
        tmp_path,
        "README.md",
        '<img src="docs/assets/readme/banner.svg" width="100%" alt="The banner">\n\n'
        "See [the guide](docs/guide.md#set-up-a-sandbox) and [the API](docs/guide.md#api).\n",
    )
    write(
        tmp_path,
        "docs/guide.md",
        "# Guide\n\n## Set up a sandbox\n\n## `api`\n\n## API\n\n"
        "Back to [the top](#guide), [the second API](#api-1) and [the README](../README.md).\n",
    )
    return tmp_path


def test_a_clean_site_passes(site: Path) -> None:
    assert check_docs.check(site) == []


def test_reports_missing_files_and_headings(site: Path) -> None:
    write(site, "docs/bad.md", "[gone](missing.md) and [no such part](guide.md#nowhere)\n")
    problems = check_docs.check(site)
    assert any("docs/bad.md:1: link target missing.md does not exist" in p for p in problems)
    assert any("no heading #nowhere in docs/guide.md" in p for p in problems)


def test_reports_undescribed_oversized_and_unused_images(site: Path) -> None:
    write(site, "docs/assets/big.png", b"\0" * (300 * 1024 + 1))
    write(site, "docs/assets/unused.svg", "<svg/>")
    write(
        site,
        "docs/pics.md",
        '<img src="assets/big.png">\n\n<picture><img src="x" alt=""></picture>\n',
    )
    problems = check_docs.check(site)
    assert any("image assets/big.png has no alt text" in p for p in problems)
    assert any("image assets/big.png is 301 KB (limit 300 KB)" in p for p in problems)
    assert any("docs/assets/unused.svg: not shown by any page" in p for p in problems)
    assert any("<picture>" in p for p in problems)


def test_code_is_not_checked(site: Path) -> None:
    write(
        site,
        "docs/code.md",
        "```\n[not a link](missing.md)\n```\n\n`[inline](missing.md)` and `<picture>`\n",
    )
    assert check_docs.check(site) == []


def test_external_links_and_link_definitions(site: Path) -> None:
    write(
        site,
        "docs/ext.md",
        "[web](https://example.com/x.md)\n\n[ref]: guide.md#guide\n[bad]: nope.md\n",
    )
    problems = check_docs.check(site)
    assert problems == ["docs/ext.md:4: link target nope.md does not exist"]


def test_github_anchor_rules() -> None:
    assert check_docs.slug("Single GPU: a tested recipe") == "single-gpu-a-tested-recipe"
    assert check_docs.slug("`agentbox bench create`") == "agentbox-bench-create"
    assert check_docs.slug("Fix / undo") == "fix--undo"
    assert check_docs.slug("[Links](x.md) and snake_case") == "links-and-snake_case"
    assert check_docs.anchors("# A\n## A\n```\n# A\n```\n") == {"a", "a-1"}
