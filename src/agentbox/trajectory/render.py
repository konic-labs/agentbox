"""Render trajectories to a self-contained HTML dashboard."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from agentbox.trajectory.schema import Trajectory


def resolve_trajectory_paths(
    source: Path,
    *,
    model_id: str | None = None,
) -> list[Path]:
    """Resolve traj JSON files from a file, dir, or bench run directory."""
    source = Path(source)
    if source.is_file() and source.suffix == ".json":
        return [source]
    if not source.is_dir():
        return []
    # bench run: models/<id>/trajectories
    models = source / "models"
    if models.is_dir():
        if model_id:
            d = models / model_id / "trajectories"
            return sorted(d.glob("*.json")) if d.is_dir() else []
        paths: list[Path] = []
        for d in sorted(models.glob("*/trajectories")):
            paths.extend(sorted(d.glob("*.json")))
        if paths:
            return paths
    # flat trajectories dir
    direct = sorted(source.glob("*.json"))
    if direct:
        return direct
    nested = sorted(source.rglob("*.json"))
    return [p for p in nested if p.name != "report.json" and "suite" not in p.parts]


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _short(s: Any, n: int = 800) -> str:
    t = "" if s is None else str(s)
    return t if len(t) <= n else t[:n] + f"\n… [{len(t) - n} more chars]"


def _is_test_related(name: str, args: dict[str, Any], result: str) -> bool:
    blob = json.dumps(args).lower() + " " + (result or "").lower()
    if name == "run_tests":
        return True
    if name == "run_command" and any(
        k in blob for k in ("pytest", "unittest", "assert ", "test_", "all tests")
    ):
        return True
    return False


def render_html(
    trajs: Sequence[Trajectory],
    out: Path,
    *,
    title: str = "AgentBox Trajectories",
    report: dict[str, Any] | None = None,
) -> Path:
    """Write a pure HTML+CSS dashboard; returns output path."""
    n = len(trajs)
    n_ok = sum(1 for t in trajs if str(t.final_status) in ("success", "FinalStatus.SUCCESS") or getattr(t.final_status, "value", None) == "success")
    # normalize status
    def status_val(t: Trajectory) -> str:
        s = t.final_status
        return s.value if hasattr(s, "value") else str(s)

    statuses = Counter(status_val(t) for t in trajs)
    n_ok = statuses.get("success", 0)
    mean_steps = sum(t.metrics.steps for t in trajs) / max(n, 1)
    mean_reward = sum(t.reward for t in trajs) / max(n, 1)

    cards: list[str] = []
    nav: list[str] = []
    for t in trajs:
        st = status_val(t)
        sc = "ok" if st == "success" else ("err" if st == "error" else "fail")
        meta = t.metadata or {}
        verify_cmd = meta.get("verify_command") or "—"
        verify_exit = meta.get("verify_exit_code")
        verify_ok = meta.get("verify_success")
        if verify_ok is None and verify_exit is not None:
            verify_ok = verify_exit == 0
        vcls = "ok" if verify_ok else ("fail" if verify_exit is not None else "unk")
        vlabel = (
            f"exit {verify_exit} · {'PASSED' if verify_ok else 'FAILED'}"
            if verify_exit is not None
            else "n/a"
        )
        anchor = re.sub(r"[^a-zA-Z0-9_-]", "_", t.task_id)
        user_msg = ""
        for m in t.messages:
            if (m.role.value if hasattr(m.role, "value") else m.role) == "user" and m.content:
                user_msg = m.content
                break

        # agent self-check
        agent_pass = None
        agent_summary = "No agent-run tests recorded"
        agent_detail = ""
        agent_cmd = "—"
        for r in reversed(t.tool_call_records):
            if _is_test_related(r.name, r.arguments, r.result):
                agent_cmd = str(r.arguments.get("command") or r.arguments)[:220]
                res = r.result or ""
                m = re.search(r"exit_code\s*=\s*(-?\d+)", res)
                if m:
                    agent_pass = int(m.group(1)) == 0
                elif "all tests passed" in res.lower() or re.search(
                    r"\d+\s+passed", res.lower()
                ):
                    agent_pass = "failed" not in res.lower().split("passed")[0][-20:] if False else (
                        not re.search(r"\d+\s+failed", res.lower())
                    )
                agent_summary = f"{r.name} @ step {r.step}"
                agent_detail = _short(res, 1200)
                break
        acls = "ok" if agent_pass is True else ("fail" if agent_pass is False else "unk")
        mismatch = ""
        if agent_pass is True and verify_ok is False:
            mismatch = (
                '<div class="mismatch">'
                "<strong>⚠ Scoring mismatch</strong> — agent self-check looked green; "
                "official suite verifier failed. Reward uses the official command only."
                "</div>"
            )

        # tools table
        rows = []
        for r in t.tool_call_records:
            unoff = _is_test_related(r.name, r.arguments, r.result)
            rows.append(
                f"<tr class='{'unoff' if unoff else ''}'>"
                f"<td>{r.step}</td><td><code>{_esc(r.name)}</code>"
                f"{' <span class=pill>self-check</span>' if unoff else ''}</td>"
                f"<td><pre>{_esc(_short(json.dumps(r.arguments), 180))}</pre></td>"
                f"<td><pre>{_esc(_short(r.result, 240))}</pre></td></tr>"
            )

        # messages
        msgs = []
        for i, m in enumerate(t.messages):
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            content = m.content or ""
            tcs = m.tool_calls or []
            body = []
            unoff = False
            if tcs:
                for tc in tcs:
                    name = tc.function.name
                    args = tc.function.arguments
                    if name in ("run_tests", "run_command") and any(
                        k in args.lower() for k in ("pytest", "assert", "python -c", "test_")
                    ):
                        unoff = True
                    body.append(
                        f"<div class='tc'><b>{_esc(name)}</b>"
                        f"{' <span class=pill>self-check</span>' if unoff else ''}"
                        f"<pre>{_esc(_short(args, 1000))}</pre></div>"
                    )
            if content:
                if unoff or (role == "tool" and any(k in content.lower() for k in ("passed", "pytest"))):
                    unoff = True
                    body.append("<div class='note'>Unofficial agent output (not scoring)</div>")
                body.append(f"<pre>{_esc(_short(content, 2000 if role != 'system' else 400))}</pre>")
            preview = (content or (tcs[0].function.name if tcs else ""))[:80]
            msgs.append(
                f"<details class='msg role-{_esc(role)}{' unoff' if unoff else ''}'>"
                f"<summary><span class='rb'>{_esc(role)}</span> #{i} "
                f"<span class='pv'>{_esc(preview)}</span></summary>"
                f"{''.join(body)}</details>"
            )

        vmark = "✓" if verify_ok else ("✗" if verify_exit is not None else "?")
        err_html = f"<p class='err'>{_esc(t.error)}</p>" if t.error else ""
        v_stdout = meta.get("verify_stdout")
        v_stderr = meta.get("verify_stderr")
        v_out_html = (
            f"<pre class='vout'>{_esc(_short(v_stdout, 1500))}</pre>" if v_stdout else ""
        )
        v_err_html = (
            f"<pre class='vout'>{_esc(_short(v_stderr, 800))}</pre>" if v_stderr else ""
        )
        agent_detail_html = (
            f"<pre class='vout'>{_esc(agent_detail)}</pre>" if agent_detail else ""
        )
        agent_pass_label = (
            "PASS" if agent_pass is True else "FAIL" if agent_pass is False else "n/a"
        )
        nav.append(
            f"<a class='nav {sc}' href='#{_esc(anchor)}'><i></i>"
            f"<b>{_esc(t.task_id)}</b><small>{_esc(st)} · verify {vmark} · {t.metrics.steps} steps</small></a>"
        )
        cards.append(
            f"<section class='card {sc}' id='{_esc(anchor)}'>"
            f"<h2><a href='#{_esc(anchor)}'>{_esc(t.task_id)}</a> "
            f"<span class='pill {sc}'>{_esc(st)}</span></h2>"
            f"<div class='stats'>"
            f"<div><s>reward</s><b>{t.reward:.0f}</b></div>"
            f"<div><s>steps</s><b>{t.metrics.steps}</b></div>"
            f"<div><s>tools</s><b>{t.metrics.tool_calls}</b></div>"
            f"<div><s>duration</s><b>{t.metrics.duration_s:.1f}s</b></div>"
            f"<div><s>official</s><b class='{vcls}'>{_esc(vlabel)}</b></div>"
            f"</div>"
            f"<p class='task'><b>Task</b> {_esc(_short(user_msg, 500))}</p>"
            f"{err_html}"
            f"<div class='vg'>"
            f"<div class='vc official {vcls}'><h3>Official suite verifier</h3>"
            f"<p>Ground truth for reward/success.</p>"
            f"<code>{_esc(verify_cmd)}</code>"
            f"<div class='kv'>exit <b>{_esc(verify_exit)}</b> · success <b>{_esc(verify_ok)}</b></div>"
            f"{v_out_html}{v_err_html}"
            f"</div>"
            f"<div class='vc agent {acls}'><h3>Agent self-check <span class='pill'>unofficial</span></h3>"
            f"<p>Mid-trajectory tests; not used for scoring.</p>"
            f"<code>{_esc(agent_cmd)}</code>"
            f"<p>{_esc(agent_summary)} · {agent_pass_label}</p>"
            f"{agent_detail_html}"
            f"</div></div>{mismatch}"
            f"<details><summary>Tool records ({len(t.tool_call_records)})</summary>"
            f"<table><thead><tr><th>step</th><th>tool</th><th>args</th><th>result</th></tr></thead>"
            f"<tbody>{''.join(rows) or '<tr><td colspan=4>none</td></tr>'}</tbody></table></details>"
            f"<details open><summary>Messages ({len(t.messages)})</summary>"
            f"<div class='tl'>{''.join(msgs)}</div></details></section>"
        )

    lb = ""
    for row in (report or {}).get("leaderboard") or []:
        lb += (
            f"<div class='lb-row'><span>{_esc(row.get('model_id'))}</span>"
            f"<b>{(row.get('success_rate') or 0)*100:.0f}%</b></div>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_esc(title)}</title>
<style>
:root{{--bg:#0b0d12;--p:#12151c;--p2:#181c26;--bd:#262b38;--tx:#e8eaef;--mu:#8b93a7;
--ok:#3dd68c;--fail:#f0a060;--err:#f07178;--ac:#6c9eff;--wn:#f5d76e;
--mono:ui-monospace,SFMono-Regular,Menlo,monospace;--sans:system-ui,-apple-system,sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;font-family:var(--sans);background:var(--bg);color:var(--tx)}}
.app{{display:grid;grid-template-columns:260px 1fr;min-height:100vh}}
@media(max-width:900px){{.app{{grid-template-columns:1fr}}}}
.side{{position:sticky;top:0;height:100vh;overflow:auto;background:var(--p);border-right:1px solid var(--bd);padding:16px 12px}}
.side h1{{font-size:13px;color:var(--mu);text-transform:uppercase;letter-spacing:.06em;margin:0 0 12px}}
.hs{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px}}
.hs div{{background:var(--p2);border:1px solid var(--bd);border-radius:8px;padding:8px}}
.hs s{{display:block;font-size:10px;color:var(--mu);text-decoration:none}}
.hs b{{font-size:18px}}.hs b.ok{{color:var(--ok)}}
.lb{{background:var(--p2);border:1px solid var(--bd);border-radius:8px;padding:8px;margin-bottom:12px;font-size:12px}}
.lb-row{{display:flex;justify-content:space-between;margin:4px 0}}
.nav{{display:block;padding:8px;border-radius:8px;text-decoration:none;color:inherit;margin:2px 0;border:1px solid transparent}}
.nav:hover{{background:var(--p2);border-color:var(--bd)}}
.nav i{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}}
.nav.ok i{{background:var(--ok)}}.nav.fail i{{background:var(--fail)}}.nav.err i{{background:var(--err)}}
.nav b{{font-size:12px;display:block}}.nav small{{color:var(--mu);font-size:10px}}
.main{{padding:24px;max-width:1100px}}
.note{{background:var(--p);border:1px solid var(--bd);border-radius:8px;padding:10px 12px;font-size:12px;color:var(--mu);margin-bottom:16px}}
.card{{background:var(--p);border:1px solid var(--bd);border-radius:12px;padding:16px;margin-bottom:16px;scroll-margin-top:16px}}
.card.ok{{border-left:3px solid var(--ok)}}.card.fail{{border-left:3px solid var(--fail)}}.card.err{{border-left:3px solid var(--err)}}
.card:target{{border-color:var(--ac)}}
.card h2{{margin:0 0 10px;font-size:16px}}.card h2 a{{color:inherit;text-decoration:none}}
.pill{{font-size:10px;padding:2px 8px;border-radius:999px;background:var(--p2);color:var(--mu);margin-left:6px}}
.pill.ok{{background:rgba(61,214,140,.15);color:var(--ok)}}.pill.fail{{background:rgba(240,160,96,.15);color:var(--fail)}}
.stats{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}}
.stats div{{background:var(--p2);border-radius:8px;padding:6px 10px}}.stats s{{display:block;font-size:9px;color:var(--mu);text-decoration:none}}
.stats b.ok{{color:var(--ok)}}.stats b.fail{{color:var(--fail)}}
.task{{font-size:13px;color:var(--mu)}}.task b{{color:var(--tx)}}
.err{{background:rgba(240,113,120,.12);color:var(--err);padding:8px;border-radius:8px;font-size:12px}}
.vg{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0}}
@media(max-width:800px){{.vg{{grid-template-columns:1fr}}}}
.vc{{border:1px solid var(--bd);border-radius:10px;padding:12px;background:var(--p2)}}
.vc.official.ok{{border-color:rgba(61,214,140,.4);background:rgba(61,214,140,.08)}}
.vc.official.fail{{border-color:rgba(240,160,96,.45);background:rgba(240,160,96,.08)}}
.vc h3{{margin:0 0 6px;font-size:12px}}.vc p{{margin:0 0 8px;font-size:11px;color:var(--mu)}}
.vc code{{display:block;font-family:var(--mono);font-size:11px;color:var(--ac);background:rgba(0,0,0,.25);padding:6px 8px;border-radius:6px;word-break:break-all}}
.kv{{font-size:12px;margin-top:8px}}.vout{{font-family:var(--mono);font-size:10px;white-space:pre-wrap;max-height:140px;overflow:auto;background:rgba(0,0,0,.3);padding:8px;border-radius:6px}}
.mismatch{{background:rgba(245,215,110,.12);border:1px solid rgba(245,215,110,.35);color:var(--wn);padding:10px;border-radius:8px;font-size:12px;margin:8px 0}}
details{{border-top:1px solid var(--bd);margin-top:10px;padding-top:8px}}
summary{{cursor:pointer;font-size:12px;font-weight:600;color:var(--mu)}}
table{{width:100%;border-collapse:collapse;font-size:11px}}
th,td{{text-align:left;padding:6px;border-bottom:1px solid var(--bd);vertical-align:top}}
th{{color:var(--mu)}}tr.unoff td{{background:rgba(245,215,110,.06)}}
pre{{margin:0;font-family:var(--mono);font-size:10px;white-space:pre-wrap;word-break:break-word;color:var(--mu)}}
.tl{{display:flex;flex-direction:column;gap:6px;margin-top:8px}}
.msg{{border:1px solid var(--bd);border-radius:8px;background:var(--p2)}}
.msg.unoff{{border-color:rgba(245,215,110,.35)}}
.msg summary{{display:flex;gap:8px;align-items:center;padding:7px 10px;font-size:11px;cursor:pointer;list-style:none}}
.rb{{font-size:9px;font-weight:700;text-transform:uppercase;padding:2px 6px;border-radius:4px;background:#2a2f3a}}
.role-user .rb{{background:rgba(108,158,255,.15);color:var(--ac)}}
.role-assistant .rb{{background:rgba(167,139,250,.15);color:#a78bfa}}
.role-tool .rb{{background:rgba(61,214,140,.12);color:var(--ok)}}
.pv{{color:var(--mu);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;font-family:var(--mono);font-size:10px}}
.tc{{padding:8px 12px;border-top:1px solid var(--bd)}}.tc b{{color:var(--ok);font-family:var(--mono);font-size:11px}}
.note{{color:var(--wn);font-size:10px;padding:8px 12px 0;font-weight:600}}
.msg pre{{padding:10px 12px;border-top:1px solid var(--bd);color:#c8cdd8;max-height:360px;overflow:auto}}
</style></head><body><div class="app">
<aside class="side">
<h1>AgentBox Trajectories</h1>
<div class="hs">
<div><s>Success</s><b class="ok">{n_ok}/{n}</b></div>
<div><s>Rate</s><b class="ok">{(100*n_ok/max(n,1)):.0f}%</b></div>
<div><s>Mean steps</s><b>{mean_steps:.1f}</b></div>
<div><s>Mean reward</s><b>{mean_reward:.2f}</b></div>
</div>
<div class="lb"><b>Leaderboard</b>{lb or f"<div class='lb-row'><span>model</span><b>{(100*n_ok/max(n,1)):.0f}%</b></div>"}</div>
{''.join(nav)}
</aside>
<main class="main">
<h1 style="margin-top:0">{_esc(title)}</h1>
<div class="note"><b>Scoring:</b> reward uses only the <b>official suite verifier</b>
(<code>verify_command</code> / <code>verify_exit_code</code>). Agent <code>run_tests</code> /
ad-hoc checks are marked unofficial.</div>
{''.join(cards)}
<p style="color:var(--mu);font-size:11px">{n} trajectories · pure HTML/CSS · generated by agentbox traj render</p>
</main></div></body></html>"""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
