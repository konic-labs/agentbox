"""art: AgentBox's README and docs art, as self-contained SVG files.

The look comes from the AgentBox release film: a fog field under a fine
halftone screen, a pale *chamber* band that opens for one rollout and prints
its container in Space Mono, the verdict by exit code, and the band shutting
into a line of red dots: the trajectory. One accent, Konic red.

Every file stands alone: text is outlined into paths (Inter Tight and Space
Mono, SIL OFL 1.1, fetched once at pinned commits and checked by SHA-256),
there is no script and no outside reference, both themes are inside the file
(prefers-color-scheme), and motion stops for readers who ask for reduced
motion. The non-animated state of every element is its final frame, so
renderers without CSS animation show the finished picture.

    python scripts/art/art.py [--out docs/assets/readme] [--fonts DIR] [--data FILE]

  banner.svg    the top of the README: the chamber opens, agentbox in red dots
  facts.svg     a strip of facts under the banner
  rollout.svg   the film: one rollout, from container to verdict to trajectory
  group.svg     four chambers of one GRPO group, each shut into its red line
  training.svg  a measured reward curve (scripts/art/training.json) and the held-out verdict
  compare.svg   the held-out tasks before and after training, from the same file
  pipeline.svg  provision -> loop -> verify -> reward -> record, for the docs
  footer.svg    the Konic mark in red strokes, konic.io
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from xml.sax.saxutils import escape, quoteattr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FONT_CACHE = os.path.join(REPO, ".cache", "art-fonts")
TRAINING_DATA = os.path.join(HERE, "training.json")
OUTPUTS = (
    "banner.svg",
    "facts.svg",
    "rollout.svg",
    "group.svg",
    "training.svg",
    "compare.svg",
    "pipeline.svg",
    "footer.svg",
)
GOOGLE_FONTS = "https://raw.githubusercontent.com/google/fonts"
FONT_FILES = {
    "InterTight[wght].ttf": (
        f"{GOOGLE_FONTS}/0b58fb370093f9a9f4ff785d94405710b79de67c/ofl/intertight/InterTight%5Bwght%5D.ttf",
        "b81b73dcb64df3c230cabade7df6c5773bf863233f24c9ee51087519f1f88b6f",
    ),
    "SpaceMono-Regular.ttf": (
        f"{GOOGLE_FONTS}/8b0a1d0f5983c89bc2b93f1b5fb55f9e252744b5/ofl/spacemono/SpaceMono-Regular.ttf",
        "95837e182baeeada83368f7748db28357f0a1b75c6b84ff7065b5edf933c8e18",
    ),
}
DISPLAY, MONO = "InterTight[wght].ttf", "SpaceMono-Regular.ttf"
RED = "#ff004d"
W = 1280

# The fog field. Light is the film's own palette (warm fog, paper bands);
# dark is the same scene at night.
THEMES = {
    "light": {
        "sky": "#f3f2ef",
        "ground": "#e3e1dc",
        "ink": "#0b0b0b",
        "muted": "#74716b",
        "faint": "#a19e97",
        "band": "#ebe9e4",
        "band-edge": "#d2cfc8",
        "rule": "#cfccc5",
        "dot": "#1a1917",
        "dot-a": "0.11",
        "fog": "#f7f6f3",
    },
    "dark": {
        "sky": "#101011",
        "ground": "#050505",
        "ink": "#f4f4f2",
        "muted": "#9b9891",
        "faint": "#5f5d58",
        "band": "#1b1b1c",
        "band-edge": "#2b2b2c",
        "rule": "#262626",
        "dot": "#ffffff",
        "dot-a": "0.07",
        "fog": "#1a1a1b",
    },
}


class ArtError(RuntimeError):
    pass


# ------------------------------------------------------------------ fonts
def ensure_fonts(fonts_dir: str | None = None) -> str:
    folder = fonts_dir or FONT_CACHE
    os.makedirs(folder, exist_ok=True)
    for name, (url, sha256) in FONT_FILES.items():
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            continue
        if fonts_dir:
            raise ArtError(f"no {name} in {fonts_dir}")
        print(f"fetching {name}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=60) as resp:
            body = resp.read()
        if hashlib.sha256(body).hexdigest() != sha256:
            raise ArtError(f"{name} from {url} does not match its pinned SHA-256")
        with open(path, "wb") as fh:
            fh.write(body)
    return folder


class Type:
    """Outlines text as SVG path data."""

    def __init__(self, folder: str) -> None:
        try:
            from fontTools.pens.svgPathPen import SVGPathPen
            from fontTools.pens.transformPen import TransformPen
            from fontTools.ttLib import TTFont
            from fontTools.varLib import instancer
        except ImportError:
            raise ArtError("outlining needs fontTools: python3 -m pip install fonttools") from None
        self._svg_pen, self._transform_pen = SVGPathPen, TransformPen
        self._open, self._instancer = TTFont, instancer
        self.folder = folder
        self._faces: dict[tuple[str, int | None], object] = {}
        self._glyph_cache: dict[str, str] = {}

    def _face(self, name: str, weight: int | None):  # noqa: ANN202
        key = (name, weight)
        if key not in self._faces:
            font = self._open(os.path.join(self.folder, name))
            if weight is not None and "fvar" in font:
                font = self._instancer.instantiateVariableFont(font, {"wght": weight})
            self._faces[key] = font
        return self._faces[key]

    def width(
        self, text: str, font: str, size: float, tracking: float = 0.0, weight: int | None = None
    ) -> float:
        return self.path(text, font, size, 0, 0, tracking, weight)[1]

    def path(
        self,
        text: str,
        font: str,
        size: float,
        x: float,
        y: float,
        tracking: float = 0.0,
        weight: int | None = None,
        anchor: str = "start",
    ) -> tuple[str, float]:
        face = self._face(font, weight)
        scale = size / face["head"].unitsPerEm
        glyphs, cmap, hmtx = face.getGlyphSet(), face.getBestCmap(), face["hmtx"]
        placed, advance = [], 0.0
        for ch in text:
            name = cmap.get(ord(ch)) or cmap.get(ord("?"))
            placed.append((name, advance))
            advance += hmtx[name][0] * scale + tracking * size
        if text:
            advance -= tracking * size
        x0 = x - {"start": 0.0, "middle": advance / 2, "end": advance}[anchor]
        parts = []
        for name, dx in placed:
            pen = self._svg_pen(glyphs, ntos=lambda v: num(round(v, 1)))
            glyphs[name].draw(self._transform_pen(pen, (scale, 0, 0, -scale, x0 + dx, y)))
            if pen.getCommands():
                parts.append(pen.getCommands())
        return " ".join(parts), advance

    def placements(
        self,
        text: str,
        font: str,
        size: float,
        x: float,
        y: float,
        tracking: float = 0.0,
        weight: int | None = None,
        anchor: str = "start",
    ) -> tuple[list[tuple[str, str, float]], float]:
        """``(glyph id, path at the origin, x)`` per visible glyph, and the advance."""
        face = self._face(font, weight)
        scale = size / face["head"].unitsPerEm
        glyphs, cmap, hmtx = face.getGlyphSet(), face.getBestCmap(), face["hmtx"]
        names, advance = [], 0.0
        for ch in text:
            name = cmap.get(ord(ch)) or cmap.get(ord("?"))
            names.append((name, advance))
            advance += hmtx[name][0] * scale + tracking * size
        if text:
            advance -= tracking * size
        x0 = x - {"start": 0.0, "middle": advance / 2, "end": advance}[anchor]
        out = []
        for name, dx in names:
            gid = f"{font[:5].lower()}{weight or ''}-{num(size)}-{name}".replace(".", "_")
            if gid not in self._glyph_cache:
                pen = self._svg_pen(glyphs, ntos=lambda v: num(round(v, 1)))
                glyphs[name].draw(self._transform_pen(pen, (scale, 0, 0, -scale, 0, 0)))
                self._glyph_cache[gid] = pen.getCommands()
            if self._glyph_cache[gid]:
                out.append((gid, self._glyph_cache[gid], x0 + dx))
        return out, advance


# The SVG being built: text() registers glyph definitions here.
_GLYPH_DEFS: dict[str, str] = {}


# ------------------------------------------------------------------ svg
def num(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def theme_css() -> str:
    light = ";".join(f"--{k}:{v}" for k, v in THEMES["light"].items())
    dark = ";".join(f"--{k}:{v}" for k, v in THEMES["dark"].items())
    return (
        f"svg{{{light}}}"
        f"@media (prefers-color-scheme:dark){{svg{{{dark}}}}}"
        ".ink{fill:var(--ink)}.muted{fill:var(--muted)}.faint{fill:var(--faint)}"
        f".red{{fill:{RED}}}.band{{fill:var(--band)}}.edge{{fill:var(--band-edge)}}"
        ".rule{fill:var(--rule)}.sky{stop-color:var(--sky)}.ground{stop-color:var(--ground)}"
        ".fogstop{stop-color:var(--fog)}"
        ".dot{fill:var(--dot);fill-opacity:var(--dot-a)}.inkline{fill:none;stroke:var(--ink)}"
        f".dots-red{{stroke:{RED}}}.dots-faint{{stroke:var(--faint)}}.dots-ink{{stroke:var(--ink)}}"
        "@media (prefers-reduced-motion:reduce){*{animation:none!important}}"
    )


@dataclass
class Svg:
    width: int
    height: int
    title: str
    defs: list[str] = field(default_factory=list)
    body: list[str] = field(default_factory=list)
    css: list[str] = field(default_factory=list)

    def add(self, *parts: str) -> None:
        self.body.extend(parts)

    def __post_init__(self) -> None:
        _GLYPH_DEFS.clear()

    def render(self) -> str:
        style = theme_css() + "".join(self.css)
        glyphs = "".join(f'<path id="{gid}" d="{d}"/>' for gid, d in _GLYPH_DEFS.items())
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}" role="img" aria-label={quoteattr(self.title)}>'
            f"<title>{escape(self.title)}</title><style>{style}</style>"
            f"<defs>{glyphs}{''.join(self.defs)}</defs>{''.join(self.body)}</svg>\n"
        )


def field_bg(svg: Svg, fog_from: float = 0.55) -> None:
    """Fog gradient under a halftone screen whose density follows the fog."""
    h = svg.height
    svg.defs.append(
        '<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" class="sky"/><stop offset="1" class="ground"/></linearGradient>'
        '<pattern id="screen" width="6" height="6" patternUnits="userSpaceOnUse">'
        '<circle cx="3" cy="3" r="0.85" class="dot"/></pattern>'
        f'<linearGradient id="fogmask" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="#fff" stop-opacity="0.9"/>'
        f'<stop offset="{num(fog_from)}" stop-color="#fff" stop-opacity="0.25"/>'
        f'<stop offset="1" stop-color="#fff" stop-opacity="0.75"/></linearGradient>'
        f'<mask id="fog"><rect width="{W}" height="{h}" fill="url(#fogmask)"/></mask>'
    )
    svg.add(
        f'<rect width="{svg.width}" height="{h}" fill="url(#sky)"/>',
        f'<rect width="{svg.width}" height="{h}" fill="url(#screen)" mask="url(#fog)"/>',
    )


def dotted_line(
    x0: float, x1: float, y: float, cls: str = "red", pitch: float = 6.0, r: float = 1.6
) -> str:
    """A line of dots: zero-length dashes with round caps."""
    return (
        f'<path d="M{num(x0)} {num(y)}H{num(x1)}" class="dots-{cls}" stroke-width="{num(2 * r)}" '
        f'stroke-dasharray="0 {num(pitch)}" stroke-linecap="round" fill="none"/>'
    )


def band(x: float, y: float, w: float, h: float, ident: str = "") -> str:
    """The chamber: a paper band whose long edges dissolve into dots."""
    edges = []
    for row, alpha in ((0, 0.55), (1, 0.3), (2, 0.14)):
        for side in (-1, 1):
            yy = y - 3 - row * 3 if side < 0 else y + h + 3 + row * 3
            edges.append(
                f'<rect x="{num(x)}" y="{num(yy - 1)}" width="{num(w)}" height="2" '
                f'fill="url(#edgedots)" fill-opacity="{alpha}"/>'
            )
    return (
        f"<g{f' id={ident!r}' if ident else ''}>"
        f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" class="band"/>'
        + "".join(edges)
        + "</g>"
    )


def edge_dots_def() -> str:
    return (
        '<pattern id="edgedots" width="4" height="2" patternUnits="userSpaceOnUse">'
        '<rect width="2" height="2" class="edge"/></pattern>'
    )


def text(
    t: Type, s: str, font: str, size: float, x: float, y: float, cls: str, **kw: object
) -> str:
    """Outlined text: each distinct glyph defined once, placed with <use>."""
    placed, _ = t.placements(s, font, size, x, y, **kw)  # type: ignore[arg-type]
    uses = []
    for gid, d, gx in placed:
        _GLYPH_DEFS.setdefault(gid, d)
        uses.append(f'<use href="#{gid}" xlink:href="#{gid}" x="{num(gx)}" y="{num(y)}"/>')
    return f'<g class="{cls}">{"".join(uses)}</g>'


def red_dot_word(
    svg: Svg, t: Type, word: str, size: float, x: float, y: float, weight: int = 500
) -> float:
    """The word drawn in red dots: outlined glyphs masking a dot screen."""
    d, width = t.path(word, DISPLAY, size, x, y, tracking=-0.02, weight=weight)
    pitch = max(4.0, size / 22)
    svg.defs.append(
        f'<pattern id="reddots" width="{num(pitch)}" height="{num(pitch)}" patternUnits="userSpaceOnUse">'
        f'<circle cx="{num(pitch / 2)}" cy="{num(pitch / 2)}" r="{num(pitch * 0.36)}" fill="{RED}"/></pattern>'
        f'<mask id="wordmask"><path d="{d}" fill="#fff"/></mask>'
    )
    svg.add(
        f'<rect x="{num(x - 4)}" y="{num(y - size)}" width="{num(width + 8)}" height="{num(size * 1.25)}" fill="url(#reddots)" mask="url(#wordmask)"/>'
    )
    return width


def rows(
    t: Type,
    x: float,
    y: float,
    items: list[tuple[str, str]],
    size: float = 15,
    gap: float = 24,
    label_w: float = 96,
    cls_value: str = "ink",
    anim: list[str] | None = None,
) -> list[str]:
    out = []
    for i, (label, value) in enumerate(items):
        yy = y + i * gap
        a = f' style="animation-delay:{anim[i]}"' if anim and i < len(anim) else ""
        out.append(
            f'<g class="row"{a}>'
            + text(t, label, MONO, size, x, yy, "muted")
            + text(t, value, MONO, size, x + label_w, yy, cls_value)
            + "</g>"
        )
    return out


# --------------------------------------------------------------- pieces
def banner(t: Type) -> str:
    svg = Svg(W, 400, "agentbox: sandboxes, rewards and exact trajectories for agent RL")
    field_bg(svg)
    svg.defs.append(edge_dots_def())
    # the chamber, opening from its centre line
    by, bh = 214, 118
    svg.css.append(
        "@keyframes open{0%{transform:scaleY(.02)}100%{transform:scaleY(1)}}"
        f"#chamber{{transform-origin:640px {by + bh / 2}px;animation:open .55s steps(6,end) .35s both}}"
        "@keyframes on{0%{opacity:0}100%{opacity:1}}"
        ".row{animation:on .01s steps(1,end) both}"
        "#title,#line,#tag{animation:on .01s steps(1,end) 1.9s both}"
    )
    svg.add(band(0, by, W, bh, "chamber"))
    items = [
        ("create", "agentbox-fix-add-3f9a1c2e"),
        ("image", "python:3.12-slim-bookworm"),
        ("network", "none"),
        ("seed", "calc.py  test_calc.py"),
    ]
    svg.add(*rows(t, 64, by + 32, items, anim=[f"{0.9 + 0.18 * i:.2f}s" for i in range(4)]))
    # verdict at the right of the band
    svg.add(
        '<g class="row" style="animation-delay:1.65s">'
        + text(t, "exit 0", MONO, 15, 1216, by + 32, "muted", anchor="end")
        + text(t, "reward 1.0", MONO, 15, 1216, by + 56, "red", anchor="end")
        + "</g>"
    )
    # title in red dots and its trajectory line
    svg.add('<g id="title">')
    width = red_dot_word(svg, t, "agentbox", 108, 60, 150)
    svg.add("</g>")
    svg.add(f'<g id="line">{dotted_line(64, 64 + width, 176)}</g>')
    svg.add(
        '<g id="tag">'
        + text(
            t,
            "Sandboxes, rewards and exact trajectories",
            DISPLAY,
            26,
            1216,
            112,
            "ink",
            anchor="end",
            weight=500,
        )
        + text(
            t,
            "for RL post-training of LLM agents.",
            DISPLAY,
            26,
            1216,
            146,
            "ink",
            anchor="end",
            weight=500,
        )
        + "</g>"
    )
    svg.add(text(t, "KONIC / AGENTBOX", MONO, 12, 64, 372, "faint", tracking=0.12))
    svg.add(
        text(
            t,
            "TRL  VERL  OPENRLHF  SKYRL  ART",
            MONO,
            12,
            1216,
            372,
            "faint",
            tracking=0.12,
            anchor="end",
        )
    )
    return svg.render()


FACTS = [
    "APACHE 2.0",
    "PYTHON 3.11+",
    "DOCKER SANDBOXES",
    "TOKEN-EXACT TRAJECTORIES",
    "pip install agentbox-rl",
]


def facts(t: Type) -> str:
    svg = Svg(W, 64, ". ".join(c if c.startswith("pip") else c.capitalize() for c in FACTS) + ".")
    svg.add(f'<rect width="{W}" height="64" class="band"/>')
    cells = FACTS
    cw = W / len(cells)
    for i, c in enumerate(cells):
        cx = cw * i + cw / 2
        cls = "red" if c.startswith("pip") else "ink"
        svg.add(
            text(
                t, c, MONO, 14, cx, 38, cls, anchor="middle", tracking=0.06 if c.isupper() else 0.0
            )
        )
        if i:
            svg.add(f'<rect x="{num(cw * i)}" y="16" width="1" height="32" class="rule"/>')
    return svg.render()


def _timeline(name: str, t_in: float, t_out: float | None, total: float) -> str:
    """Keyframes: hidden until t_in, shown until t_out (or the end)."""
    a = 100 * t_in / total
    b = 100 * (t_out / total) if t_out is not None else 100
    eps = 0.01
    frames = ["0%{opacity:0}", f"{num(max(0, a - eps))}%{{opacity:0}}", f"{num(a)}%{{opacity:1}}"]
    if t_out is not None:
        frames += [f"{num(b - eps)}%{{opacity:1}}", f"{num(b)}%{{opacity:0}}", "100%{opacity:0}"]
    else:
        frames.append("100%{opacity:1}")
    return f"@keyframes {name}{{{''.join(frames)}}}"


def rollout(t: Type) -> str:
    """The film: create -> tools -> verdict -> the band shuts into the trajectory."""
    total = 14.0
    svg = Svg(
        W,
        520,
        "One AgentBox rollout: a fresh container is created and seeded, the agent reads, edits and tests, the verifier runs pytest and exits 0, reward 1.0, then the sandbox is destroyed and its trajectory recorded.",
    )
    field_bg(svg, 0.45)
    svg.defs.append(edge_dots_def())
    by, bh = 118, 300
    css = [
        "@keyframes openshut{0%{transform:scaleY(.01)}5%{transform:scaleY(1)}86%{transform:scaleY(1)}"
        "90%{transform:scaleY(.01)}100%{transform:scaleY(.01)}}",
        f"#chamber{{transform-origin:640px {by + bh / 2}px;animation:openshut {total}s steps(48,end) infinite}}",
    ]
    steps: list[tuple[str, float, float | None, str]] = []  # (svg, t_in, t_out, id)
    head = [
        ("create", "agentbox-fix-add-3f9a1c2e"),
        ("image", "python:3.12-slim-bookworm"),
        ("network", "none"),
        ("seed", "calc.py  test_calc.py"),
    ]
    for i, (label, value) in enumerate(head):
        y = by + 46 + i * 28
        steps.append(
            (
                text(t, label, MONO, 16, 72, y, "muted") + text(t, value, MONO, 16, 192, y, "ink"),
                0.9 + 0.3 * i,
                4.2,
                f"h{i}",
            )
        )
    tools = [
        ("1", "read_file   calc.py", "return a - b", "muted"),
        ("2", "read_file   ../../etc/passwd", "path escapes workspace", "red"),
        ("3", "run_tests", "1 failed, 2 passed", "muted"),
        ("4", "write_file  calc.py", "written", "muted"),
        ("5", "run_tests", "3 passed", "muted"),
    ]
    for i, (n, call, res, cls) in enumerate(tools):
        y = by + 46 + i * 28
        steps.append(
            (
                text(t, n, MONO, 16, 72, y, "faint")
                + text(t, call, MONO, 16, 112, y, "ink")
                + text(t, res, MONO, 16, 1208, y, cls, anchor="end"),
                4.4 + 0.6 * i,
                8.4,
                f"t{i}",
            )
        )
    steps.append(
        (
            text(t, "verify", MONO, 16, 72, by + 46, "muted")
            + text(t, "python -m pytest -q", MONO, 16, 192, by + 46, "ink"),
            8.6,
            12.0,
            "v0",
        )
    )
    steps.append((text(t, "exit 0", MONO, 64, 72, by + 168, "ink"), 9.4, 12.0, "v1"))
    steps.append(
        (text(t, "reward 1.0", MONO, 64, 1208, by + 168, "red", anchor="end"), 10.0, 12.0, "v2")
    )
    steps.append(
        (
            text(t, "failure_kind none   steps 5   tool_calls 5", MONO, 16, 72, by + 250, "muted"),
            10.6,
            12.0,
            "v3",
        )
    )
    # captions under the band
    caps = [
        ("Built clean for one rollout.", 0.6, 4.2),
        ("Nothing it runs touches the host.", 4.4, 8.4),
        ("The exit code decides.", 8.6, 12.0),
        ("Then the world is destroyed; the trajectory stays.", 12.2, None),
    ]
    for i, (cap, a, b) in enumerate(caps):
        steps.append((text(t, cap, DISPLAY, 28, 72, 84, "ink", weight=500), a, b, f"c{i}"))
    svg.add(band(0, by, W, bh, "chamber"))
    for i, (frag, a, b, ident) in enumerate(steps):
        name = f"k{i}"
        css.append(_timeline(name, a, b, total))
        css.append(f"#{ident}{{animation:{name} {total}s linear infinite}}")
        svg.add(f'<g id="{ident}">{frag}</g>')
    # the trajectory appears as the band shuts
    css.append(_timeline("traj", 12.5, None, total))
    css.append(f"#traj{{animation:traj {total}s linear infinite}}")
    svg.add(f'<g id="traj">{dotted_line(0, W, by + bh / 2, pitch=6, r=1.8)}</g>')
    svg.add(
        text(t, "ROLLOUT 1/1", MONO, 12, 72, 492, "faint", tracking=0.12)
        + text(t, "AGENTBOX", MONO, 12, 1208, 492, "faint", tracking=0.12, anchor="end")
    )
    # base styles = the verdict frame: what renderers without CSS animation,
    # and readers who ask for reduced motion, see
    css.append("#h0,#h1,#h2,#h3,#t0,#t1,#t2,#t3,#t4,#c0,#c1,#c3,#traj{opacity:0}")
    svg.css.extend(css)
    return svg.render()


def group(t: Type) -> str:
    svg = Svg(
        W,
        380,
        "A GRPO group: the same task in four sandboxes, rewards 1.0, 0.0, 1.0 and 1.0; each sandbox shuts into its red trajectory line.",
    )
    field_bg(svg, 0.5)
    svg.defs.append(edge_dots_def())
    svg.add(text(t, "Grouped for GRPO.", DISPLAY, 28, 72, 60, "ink", weight=500))
    svg.add(
        text(
            t,
            "One task, four worlds, four rewards: the group's spread is the learning signal.",
            DISPLAY,
            18,
            72,
            92,
            "muted",
            weight=450,
        )
    )
    rewards = [("1.0", 5), ("0.0", 6), ("1.0", 4), ("1.0", 5)]
    vals = [float(r) for r, _ in rewards]
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    for i, (r, steps) in enumerate(rewards):
        y = 128 + i * 60
        svg.add(band(0, y, W, 36))
        svg.add(text(t, f"rollout {i + 1}/4", MONO, 15, 72, y + 23, "muted"))
        svg.add(text(t, f"steps {steps}", MONO, 15, 260, y + 23, "ink"))
        svg.add(text(t, "fix_add_001", MONO, 15, 420, y + 23, "faint"))
        adv = (float(r) - mean) / (std + 1e-6)
        svg.add(text(t, f"advantage {adv:+.2f}", MONO, 15, 1030, y + 23, "muted", anchor="end"))
        svg.add(
            text(
                t,
                f"reward {r}",
                MONO,
                15,
                1208,
                y + 23,
                "red" if r != "0.0" else "ink",
                anchor="end",
            )
        )
        svg.add(dotted_line(560, 860, y + 18, cls="red" if r != "0.0" else "faint", pitch=6, r=1.5))
    return svg.render()


def smooth(points: list[list[float]], k: int) -> list[tuple[float, float]]:
    """Trailing mean over the last ``k`` points (fewer at the start)."""
    out = []
    for i, (s, _) in enumerate(points):
        window = [v for _, v in points[max(0, i - k + 1) : i + 1]]
        out.append((s, sum(window) / len(window)))
    return out


def training(t: Type, data: dict) -> str:
    """Measured reward curves (the first run in ink, replicates fainter) and the verdict."""
    runs, verdict = data["runs"], data["eval"]
    svg = Svg(W, 460, data["title"])
    field_bg(svg, 0.6)
    svg.defs.append(edge_dots_def())
    x0, x1, y0, y1 = 112, 840, 372, 140  # plot box, y0 at the bottom
    max_step = max(s for run in runs for s, _ in run["train"])
    sx = lambda s: x0 + (x1 - x0) * s / max_step  # noqa: E731
    sy = lambda r: y0 - (y0 - y1) * r  # noqa: E731
    svg.add(text(t, data["heading"], DISPLAY, 28, 72, 60, "ink", weight=500))
    svg.add(text(t, data["subheading"], DISPLAY, 17, 72, 90, "muted", weight=450))
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        svg.add(f'<rect x="{x0}" y="{num(sy(r))}" width="{x1 - x0}" height="1" class="rule"/>')
        svg.add(text(t, f"{r:.2f}", MONO, 12, x0 - 14, sy(r) + 4, "faint", anchor="end"))
    for s in range(0, max_step + 1, 5):
        svg.add(text(t, str(s), MONO, 12, sx(s), y0 + 24, "faint", anchor="middle"))
    svg.add(text(t, "STEP", MONO, 12, x1, y0 + 46, "faint", anchor="end", tracking=0.12))
    for i, run in enumerate(reversed(runs)):  # the first run draws last, on top
        first = i == len(runs) - 1
        if first:
            for s, r in run["train"]:
                svg.add(f'<circle cx="{num(sx(s))}" cy="{num(sy(r))}" r="2.4" class="faint"/>')
        pts = " ".join(f"{num(sx(s))},{num(sy(r))}" for s, r in smooth(run["train"], 4))
        line = (
            'class="inkline" stroke-width="1.6"'
            if first
            else ('class="inkline" stroke-width="1.2" stroke-opacity="0.35"')
        )
        svg.add(f'<polyline points="{pts}" {line}/>')
        vpts = " ".join(f"{num(sx(s))},{num(sy(r))}" for s, r in run["val"])
        alpha = "" if first else ' opacity="0.4"'
        svg.add(
            f'<polyline points="{vpts}" class="dots-red" stroke-width="2.6" '
            f'stroke-dasharray="0 7" stroke-linecap="round" fill="none"{alpha}/>'
        )
        for s, r in run["val"]:
            svg.add(
                f'<circle cx="{num(sx(s))}" cy="{num(sy(r))}" r="{5 if first else 3.5}" class="red"{alpha}/>'
            )
    # legend, one row above the plot
    lx, ly = x0, 124
    svg.add(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" class="red"/>')
    svg.add(text(t, data["legend_val"], MONO, 13, lx + 18, ly, "ink"))
    lx2 = lx + 18 + t.width(data["legend_val"], MONO, 13) + 36
    svg.add(f'<rect x="{num(lx2)}" y="{ly - 5}" width="14" height="2" class="ink"/>')
    svg.add(text(t, data["legend_train"], MONO, 13, lx2 + 22, ly, "ink"))
    if len(runs) > 1:
        lx3 = lx2 + 22 + t.width(data["legend_train"], MONO, 13) + 36
        svg.add(
            f'<rect x="{num(lx3)}" y="{ly - 5}" width="14" height="2" class="ink" fill-opacity="0.35"/>'
        )
        svg.add(text(t, data["legend_replicate"], MONO, 13, lx3 + 22, ly, "muted"))
    # the verdict: held-out success before and after, as in the film's chamber
    px, pw = 896, W - 72 - 896
    svg.add(band(px, y1, pw, y0 - y1))
    left, right = px + 24, px + pw - 24
    svg.add(text(t, verdict["label"], MONO, 12, left, y1 + 34, "faint", tracking=0.12))
    svg.add(text(t, verdict["base_label"], MONO, 13, left, y1 + 78, "muted"))
    svg.add(
        text(t, f"{verdict['base'] * 100:.1f}%", MONO, 26, right, y1 + 80, "muted", anchor="end")
    )
    svg.add(text(t, verdict["trained_label"], MONO, 13, left, y1 + 118, "muted"))
    svg.add(
        text(t, f"{verdict['trained'] * 100:.1f}%", MONO, 56, right, y1 + 176, "red", anchor="end")
    )
    svg.add(dotted_line(left, right, y1 + 190, pitch=6, r=1.4))
    for i, line in enumerate(verdict["note"]):
        svg.add(text(t, line, MONO, 12, left, y1 + 210 + 16 * i, "faint"))
    svg.add(text(t, data["footnote"], MONO, 12, 72, 444, "faint"))
    return svg.render()


def compare(t: Type, data: dict) -> str:
    """Per-task successes before and after training: a faint line of dots, then a red one."""
    tasks, verdict = data["tasks"], data["eval"]
    pitch, top = 40, 128
    height = top + pitch * len(tasks) + 56
    svg = Svg(W, height, data["compare_title"])
    field_bg(svg, 0.5)
    svg.defs.append(edge_dots_def())
    svg.add(text(t, data["compare_heading"], DISPLAY, 28, 72, 60, "ink", weight=500))
    svg.add(text(t, data["compare_subheading"], DISPLAY, 17, 72, 90, "muted", weight=450))
    x0, span = 340, 560  # the track: 0% at x0, 100% at x0 + span
    for i, task in enumerate(tasks):
        y = top + i * pitch
        svg.add(band(0, y, W, 28))
        svg.add(text(t, task["id"], MONO, 14, 72, y + 19, "muted"))
        n, base, after = task["n"], task["base"], task["trained"]  # successes out of n
        if base:
            svg.add(dotted_line(x0, x0 + span * base / n, y + 10, cls="faint", pitch=6, r=1.5))
        if after:
            svg.add(dotted_line(x0, x0 + span * after / n, y + 18, cls="red", pitch=6, r=1.5))
        delta = after - base
        svg.add(text(t, f"{base}/{n} → {after}/{n}", MONO, 14, 1080, y + 19, "muted", anchor="end"))
        cls = "red" if delta > 0 else ("ink" if delta < 0 else "faint")
        svg.add(text(t, f"{delta:+d}" if delta else "0", MONO, 14, 1208, y + 19, cls, anchor="end"))
    foot = top + pitch * len(tasks) + 24
    svg.add(text(t, "BASE", MONO, 12, 72, foot, "faint", tracking=0.12))
    svg.add(dotted_line(122, 170, foot - 4, cls="faint", pitch=6, r=1.5))
    after_label = verdict["trained_label"].upper()
    svg.add(text(t, after_label, MONO, 12, 200, foot, "faint", tracking=0.12))
    after_w = t.width(after_label, MONO, 12, tracking=0.12)
    svg.add(
        dotted_line(200 + after_w + 16, 200 + after_w + 64, foot - 4, cls="red", pitch=6, r=1.5)
    )
    svg.add(
        text(
            t,
            verdict["note"][0].upper(),
            MONO,
            12,
            1208,
            foot,
            "faint",
            tracking=0.12,
            anchor="end",
        )
    )
    return svg.render()


def pipeline(t: Type) -> str:
    svg = Svg(
        W,
        300,
        "One rollout: provision a fresh sandbox, run the agent loop, verify, compute the reward, record the trajectory.",
    )
    field_bg(svg, 0.5)
    svg.defs.append(edge_dots_def())
    stages = [
        ("01", "provision", "fresh, seeded sandbox"),
        ("02", "agent loop", "model and tools"),
        ("03", "verify", "tamper checks, exit"),
        ("04", "reward", "weighted components"),
        ("05", "record", "trajectory, tokens"),
    ]
    cw = (W - 144) / len(stages)
    by = 92
    svg.add(band(72, by, W - 144, 120))
    for i, (n, name, note) in enumerate(stages):
        x = 72 + i * cw
        if i:
            svg.add(f'<rect x="{num(x)}" y="{by + 16}" width="1" height="88" class="rule"/>')
        svg.add(text(t, n, MONO, 13, x + 22, by + 34, "faint"))
        svg.add(text(t, name, DISPLAY, 24, x + 22, by + 70, "ink", weight=500))
        svg.add(text(t, note, MONO, 12, x + 22, by + 96, "muted"))
    svg.add(dotted_line(72, W - 72, by + 150, pitch=6, r=1.6))
    svg.add(
        text(
            t,
            "the sandbox is destroyed; the red line is what remains",
            MONO,
            13,
            W - 72,
            by + 178,
            "muted",
            anchor="end",
        )
    )
    svg.add(text(t, "One rollout", DISPLAY, 26, 72, 58, "ink", weight=500))
    return svg.render()


MARK_D = (
    "M 169.748 476.331 L 403.686 71.138 A 16.384 16.384 0 0 1 428.944 67.25 A 256 256 0 0 1 179.01 500.149 "
    "A 16.384 16.384 0 0 1 169.748 476.331 Z M 342.252 35.669 L 108.314 440.862 A 16.384 16.384 0 0 1 83.056 444.75 "
    "A 256 256 0 0 1 332.99 11.851 A 16.384 16.384 0 0 1 342.252 35.669 Z"
)


def footer(t: Type) -> str:
    svg = Svg(W, 220, "Konic. konic.io")
    field_bg(svg, 0.4)
    size = 92
    scale = size / 512
    cx, cy = 640, 92
    # the mark drawn in thirteen red strokes, as at the end of the film
    stripes = "".join(
        f'<rect x="0" y="{num(i * 512 / 13)}" width="512" height="{num(512 / 13 * 0.58)}" fill="#fff"/>'
        for i in range(13)
    )
    svg.defs.append(f'<mask id="strokes"><g>{stripes}</g></mask>')
    svg.add(
        f'<g transform="translate({num(cx - size / 2)} {num(cy - size / 2)}) scale({num(scale)})">'
        f'<path d="{MARK_D}" fill="{RED}" mask="url(#strokes)"/></g>'
    )
    svg.add(text(t, "konic.io", DISPLAY, 26, 640, 182, "ink", anchor="middle", weight=500))
    return svg.render()


def write_all(out: str, fonts: str | None, data_path: str | None = None) -> list[str]:
    t = Type(ensure_fonts(fonts))
    os.makedirs(out, exist_ok=True)
    with open(data_path or TRAINING_DATA, encoding="utf-8") as fh:
        data = json.load(fh)
    pieces = {
        "banner.svg": banner(t),
        "facts.svg": facts(t),
        "rollout.svg": rollout(t),
        "group.svg": group(t),
        "training.svg": training(t, data),
        "compare.svg": compare(t, data),
        "pipeline.svg": pipeline(t),
        "footer.svg": footer(t),
    }
    assert tuple(pieces) == OUTPUTS
    written = []
    for name, body in pieces.items():
        path = os.path.join(out, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        written.append(f"{path} {len(body) // 1024} KB")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "assets", "readme"))
    ap.add_argument("--fonts")
    ap.add_argument("--data", help=f"training curves and verdict (default {TRAINING_DATA})")
    args = ap.parse_args()
    try:
        for line in write_all(args.out, args.fonts, args.data):
            print(line)
    except ArtError as exc:
        print(f"art: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
