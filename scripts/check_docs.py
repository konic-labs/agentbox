"""Check the README and the docs: links resolve, and images are present, small and described.

    python scripts/check_docs.py      # exit 1 and print every problem

- Every relative link in README.md and docs/**/*.md points at a file or directory
  that exists, and a ``#fragment`` names a heading of the target page (GitHub's
  anchor rules, duplicates numbered).
- Every image has alt text (``alt=""`` marks a decorative one), is under
  300 KB, and is shown with a plain ``<img>`` or a Markdown image: GitHub's mobile
  apps leave a ``<picture>`` out, image and all.
- Every file under docs/assets/ is shown by some page.

Links inside code blocks and inline code are not links, and are skipped.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
MAX_IMAGE_BYTES = 300 * 1024

FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
MD_LINK = re.compile(r"(!?)\[((?:[^\[\]]|\[[^\]]*\])*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
LINK_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+\"[^\"]*\")?\s*$")
IMG_TAG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
PICTURE = re.compile(r"<picture\b", re.IGNORECASE)
ATTR = re.compile(r"""(\w+)\s*=\s*("[^"]*"|'[^']*')""")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.IGNORECASE)


def pages(root: Path) -> list[Path]:
    found = [root / "README.md"] if (root / "README.md").is_file() else []
    return found + sorted((root / "docs").rglob("*.md"))


def prose_lines(text: str) -> Iterable[tuple[int, str]]:
    """Lines outside fenced code, with inline code blanked out."""
    fenced = False
    for number, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            yield number, INLINE_CODE.sub(lambda m: " " * len(m.group()), line)


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: markup dropped, lowercased, punctuation removed."""
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(text: str) -> set[str]:
    seen: dict[str, int] = {}
    out = set()
    fenced = False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        m = None if fenced else HEADING.match(line)
        if not m:
            continue
        base = slug(m.group(2))
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.add(base if n == 0 else f"{base}-{n}")
    for m in re.finditer(r"""<a\s+(?:name|id)=["']([^"']+)["']""", text):
        out.add(m.group(1))
    return out


def references(text: str) -> Iterable[tuple[int, str, str, str | None]]:
    """``(line, kind, target, alt)`` for every link and image on a page."""
    for number, line in prose_lines(text):
        for m in MD_LINK.finditer(line):
            is_image = m.group(1) == "!"
            yield (
                number,
                "image" if is_image else "link",
                m.group(3),
                m.group(2) if is_image else None,
            )
        for tag in IMG_TAG.findall(line):
            attrs = {k.lower(): v[1:-1] for k, v in ATTR.findall(tag)}
            if "src" in attrs:
                yield number, "image", attrs["src"], attrs.get("alt")
        definition = LINK_DEF.match(line)
        if definition:
            yield number, "link", definition.group(1), None


def check(root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    shown: set[Path] = set()
    anchor_cache: dict[Path, set[str]] = {}
    for page in pages(root):
        text = page.read_text(encoding="utf-8")
        where = page.relative_to(root)
        if any(PICTURE.search(line) for _, line in prose_lines(text)):
            problems.append(f"{where}: <picture> is dropped by GitHub's mobile apps; use <img>")
        for number, kind, target, alt in references(text):
            if EXTERNAL.match(target):
                continue
            path_part, _, fragment = target.partition("#")
            path_part = unquote(path_part.split("?", 1)[0])
            dest = (page.parent / path_part).resolve() if path_part else page
            at = f"{where}:{number}"
            if not dest.exists():
                problems.append(f"{at}: {kind} target {target} does not exist")
                continue
            if kind == "image":
                shown.add(dest)
                if alt is None:
                    problems.append(f"{at}: image {target} has no alt text")
                if dest.is_file() and dest.stat().st_size > MAX_IMAGE_BYTES:
                    size = -(-dest.stat().st_size // 1024)
                    problems.append(f"{at}: image {target} is {size} KB (limit 300 KB)")
            if fragment and dest.is_file() and dest.suffix == ".md":
                if dest not in anchor_cache:
                    anchor_cache[dest] = anchors(dest.read_text(encoding="utf-8"))
                if unquote(fragment).lower() not in anchor_cache[dest]:
                    problems.append(f"{at}: no heading #{fragment} in {dest.relative_to(root)}")
    assets = root / "docs" / "assets"
    if assets.is_dir():
        for asset in sorted(p for p in assets.rglob("*") if p.is_file()):
            if asset.resolve() not in shown:
                problems.append(f"{asset.relative_to(root)}: not shown by any page")
    return problems


def main() -> int:
    problems = check()
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"docs ok: {len(pages(ROOT))} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
