"""Render SVGs to PNG previews (light and dark, chosen animation times) for visual QA.

Runs inside mcr.microsoft.com/playwright/python:
    python render.py IN_DIR OUT_DIR [name.svg@seconds ...]
"""

import pathlib
import sys

from playwright.sync_api import sync_playwright

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
times: dict[str, list[float]] = {}
for spec in sys.argv[3:]:
    name, _, t = spec.partition("@")
    times.setdefault(name, []).append(float(t))

with sync_playwright() as p:
    browser = p.chromium.launch()
    for svg in sorted(src.glob("*.svg")):
        for scheme in ("light", "dark"):
            for t in times.get(svg.name, [None]):  # None: let every animation finish
                page = browser.new_page(color_scheme=scheme, device_scale_factor=1)
                body = svg.read_text()
                page.set_content(
                    f'<html><body style="margin:0;background:#888">{body}</body></html>'
                )
                if t is None:
                    page.wait_for_timeout(2600)
                else:  # pause every animation at t seconds
                    page.evaluate(
                        """t => { for (const a of document.getAnimations()) {
                               a.pause(); a.currentTime = t * 1000; } }""",
                        t,
                    )
                el = page.query_selector("svg")
                suffix = "" if t is None else f"-{t:g}s"
                el.screenshot(path=str(out / f"{svg.stem}-{scheme}{suffix}.png"))
                page.close()
    browser.close()
print("rendered", len(list(out.glob("*.png"))), "png")
