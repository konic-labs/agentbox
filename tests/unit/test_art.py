"""scripts/art/art.py: the README and docs art.

The helpers run without fontTools or the fonts. Drawing every piece needs both;
that test runs where the fonts are cached (python scripts/art/art.py fetches them).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "agentbox_docs_art",  # not "art": that name belongs to OpenPipe ART
    REPO / "scripts" / "art" / "art.py",
)
assert _SPEC and _SPEC.loader
art = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = art  # dataclasses resolve annotations through sys.modules
_SPEC.loader.exec_module(art)


def test_numbers_are_short_and_never_negative_zero() -> None:
    assert art.num(3.0) == "3"
    assert art.num(2.456) == "2.46"
    assert art.num(-0.001) == "0"


def test_a_dotted_line_is_zero_length_dashes_with_round_caps() -> None:
    line = art.dotted_line(10, 70, 5, pitch=6, r=1.5)
    assert 'd="M10 5H70"' in line
    assert 'stroke-dasharray="0 6"' in line and 'stroke-linecap="round"' in line
    assert 'stroke-width="3"' in line


def test_both_themes_and_reduced_motion_are_in_the_css() -> None:
    css = art.theme_css()
    assert "@media (prefers-color-scheme:dark)" in css
    assert "@media (prefers-reduced-motion:reduce){*{animation:none!important}}" in css
    for name in art.THEMES["light"]:
        assert f"--{name}:" in css


def test_a_timeline_hides_then_shows_then_hides() -> None:
    frames = art._timeline("k", 2.0, 6.0, 10.0)
    assert frames.startswith("@keyframes k{0%{opacity:0}")
    assert "20%{opacity:1}" in frames and "60%{opacity:0}" in frames
    assert art._timeline("k", 5.0, None, 10.0).endswith("100%{opacity:1}}")


def test_a_rendered_svg_is_one_titled_image() -> None:
    svg = art.Svg(100, 50, "A test & its title")
    svg.add('<rect width="100" height="50" class="band"/>')
    root = ET.fromstring(svg.render())
    assert root.get("role") == "img"
    assert root.get("viewBox") == "0 0 100 50"
    assert root.find("{http://www.w3.org/2000/svg}title").text == "A test & its title"


def test_the_trailing_mean_smooths_the_curve() -> None:
    assert art.smooth([[1, 0.0], [2, 1.0], [3, 1.0], [4, 0.0]], 2) == [
        (1, 0.0),
        (2, 0.5),
        (3, 1.0),
        (4, 0.5),
    ]


def test_the_dither_is_konic_websites_math() -> None:
    # the Bayer table and hash of konic-website-v2's DitherBackground
    assert art.bayer8(0, 0) == 0.5 / 64
    assert art.bayer8(1, 0) == 32.5 / 64 and art.bayer8(0, 1) == 48.5 / 64
    assert art.bayer8(8, 16) == art.bayer8(0, 0)
    assert 0 <= art.hash21(3.7, 9.1) < 1


def test_the_dither_field_stays_clear_of_the_text() -> None:
    dots = art.dither_dots(1280, 636, coverage=0.3, amplitude=0.1, cell=14)
    assert dots == art.dither_dots(1280, 636, coverage=0.3, amplitude=0.1, cell=14)  # still
    assert max(x + size for x, _, size, _, _ in dots) < art.OG_TEXT_X - 64
    reds = [alpha for *_, red, alpha in dots if red]
    blacks = [alpha for *_, red, alpha in dots if not red]
    assert reds and blacks and max(blacks) <= 0.2 + 1e-9  # black dots at 20 % at most


def test_the_training_data_is_a_measured_curve() -> None:
    data = json.loads(Path(art.TRAINING_DATA).read_text())
    for run in data["runs"]:
        steps = [s for s, _ in run["train"]]
        assert steps == sorted(steps) and len(steps) >= 10
        assert all(0.0 <= r <= 1.0 for _, r in run["train"] + run["val"])
    assert 0.0 <= data["eval"]["base"] < data["eval"]["trained"] <= 1.0
    assert data["tasks"] and all(
        0 <= t[k] <= t["n"] for t in data["tasks"] for k in ("base", "trained")
    )


def test_every_readme_image_has_a_generator() -> None:
    shipped = sorted(p.name for p in (REPO / "docs" / "assets" / "readme").glob("*.svg"))
    assert shipped == sorted(art.OUTPUTS)


def _fonts_cached() -> bool:
    return all(os.path.isfile(os.path.join(art.FONT_CACHE, name)) for name in art.FONT_FILES)


@pytest.mark.skipif(not _fonts_cached(), reason="fonts not cached: run scripts/art/art.py once")
def test_every_piece_draws_a_small_valid_svg(tmp_path: Path) -> None:
    pytest.importorskip("fontTools")
    art.write_all(str(tmp_path), None, art.TRAINING_DATA)
    preview = ET.fromstring(art.og(art.Type(art.ensure_fonts())))
    assert preview.get("viewBox") == "0 0 1280 640"  # GitHub's social preview size
    for name in art.OUTPUTS:
        body = (tmp_path / name).read_text()
        root = ET.fromstring(body)
        assert root.find("{http://www.w3.org/2000/svg}title").text
        assert len(body.encode()) < 300 * 1024, name
        external = body.replace("http://www.w3.org/", "")
        assert "<script" not in body and "http://" not in external, name
