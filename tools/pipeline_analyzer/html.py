"""Assemble the self-contained, theme-aware HTML report."""
from __future__ import annotations

import html as _html
import statistics
from datetime import datetime

from pathlib import Path

from .diff import DagDiff, diff_dags
from .lineage import Lineage
from .merged import build_merged
from .render import render_dag, diff_status_map, render_lineage

_CSS = """
:root{--bg:#f7f8fa;--fg:#1a1d24;--muted:#5b6472;--card:#ffffff;--border:#e2e6ec;
--accent:#2563eb;--add:#15a34a;--rem:#dc2626;--flat:#b7791f;--chip:#eef1f5;--canvas:#ffffff;}
:root[data-theme=dark]{--bg:#0f1319;--fg:#e6e9ef;--muted:#9aa4b2;--card:#161b22;--border:#2a323d;
--accent:#5b9bff;--add:#3fb950;--rem:#f85149;--flat:#d9a441;--chip:#1e2530;--canvas:#f2f4f7;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f1319;--fg:#e6e9ef;--muted:#9aa4b2;
--card:#161b22;--border:#2a323d;--accent:#5b9bff;--add:#3fb950;--rem:#f85149;--flat:#d9a441;--chip:#1e2530;--canvas:#f2f4f7;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--border)}
h3{font-size:14px;margin:18px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.sub{color:var(--muted);margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 18px;margin:14px 0}
.canvas{background:var(--canvas);border:1px solid var(--border);border-radius:8px;padding:10px;overflow:auto;text-align:center}
.canvas svg{max-width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:10px 0;font-size:12px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:13px;height:13px;border-radius:3px;border:1px solid #888;display:inline-block}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}
.chip{background:var(--chip);border:1px solid var(--border);border-radius:20px;padding:2px 10px;font-size:12px;white-space:nowrap}
.chip.add{color:var(--add);border-color:var(--add)}.chip.rem{color:var(--rem);border-color:var(--rem)}
.score{font-size:22px;font-weight:600}
.delta.up{color:var(--add)}.delta.down{color:var(--rem)}.delta.flat{color:var(--flat)}
.meta{display:flex;flex-wrap:wrap;gap:18px 30px;align-items:baseline}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:760px){.grid2{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid var(--border);padding:5px 8px;text-align:left}
th{color:var(--muted);font-weight:600}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
td.mono{overflow-wrap:anywhere}
.pill{display:inline-block;font-size:11px;padding:1px 8px;border-radius:20px;border:1px solid var(--border);color:var(--muted)}
.err{color:var(--rem)}
.muted{color:var(--muted)}
ul.ops{margin:6px 0;padding-left:18px}ul.ops li{margin:2px 0}
.arrow{color:var(--muted)}
.num{text-align:right;font-variant-numeric:tabular-nums}th.num{text-align:right}
.rt{display:flex;flex-wrap:wrap;gap:6px 22px;margin:2px 0 10px;font-size:13px}
.rt b{font-variant-numeric:tabular-nums;font-weight:600}
.rt .k{color:var(--muted)}
.memchart{width:100%;height:96px;display:block;margin:4px 0 2px}
.memchart .area{fill:var(--accent);opacity:.16}
.memchart .line{fill:none;stroke:var(--accent);stroke-width:1.6}
.memchart .scored{fill:var(--fg);opacity:.05}
.memchart .peak{stroke:var(--rem);stroke-width:1;stroke-dasharray:3 3}
.memchart text{fill:var(--muted);font-size:10px}
details.dagbox{margin:8px 0}
details.dagbox>summary{list-style:none;cursor:pointer;user-select:none;display:flex;align-items:center;gap:6px;
 font-size:14px;margin:18px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
details.dagbox>summary::-webkit-details-marker{display:none}
details.dagbox>summary::before{content:"\\25B8";display:inline-block;transition:transform .15s ease;color:var(--muted)}
details.dagbox[open]>summary::before{transform:rotate(90deg)}
details.dagbox>summary:hover{color:var(--fg)}
"""

def _asset(name: str) -> str:
    """Read an inlined asset (``explorer.css`` / ``explorer.js``) shipped beside
    this module. They live in their own files rather than as Python strings so
    they stay editable as CSS/JS."""
    return (Path(__file__).parent / name).read_text(encoding="utf-8")


def esc(x):
    return _html.escape(str(x)) if x is not None else ""


def _delta_span(delta, improvement=None):
    """The raw delta, coloured by whether it was an *improvement* -- which is not
    the same sign when the metric is one where lower is better."""
    if delta is None:
        return '<span class="delta flat">—</span>'
    up = delta if improvement is None else improvement
    cls = "up" if up > 0.0005 else "down" if up < -0.0005 else "flat"
    sign = "+" if delta >= 0 else ""
    return f'<span class="delta {cls}">{sign}{delta:.4f}</span>'


def _hist_delta_html(diff: DagDiff):
    rows = []
    for t, (pc, cc) in diff.hist_delta.items():
        if pc == cc:
            change = f'<span class="muted">{cc}</span>'
        else:
            arrow = "▲" if cc > pc else "▼"
            cls = "add" if cc > pc else "rem"
            change = f'{pc} <span class="arrow">→</span> <span class="chip {cls}">{cc} {arrow}</span>'
        rows.append(f"<tr><td class=mono>{esc(t)}</td><td>{change}</td></tr>")
    return "<table><tr><th>operator</th><th>count</th></tr>" + "".join(rows) + "</table>"


def _est_delta_html(diff: DagDiff):
    if not diff.estimator_deltas:
        return ""
    blocks = []
    for d in diff.estimator_deltas:
        if d.kind == "swap":
            head = f'<b>estimator swap</b>: <code>{esc(d.old_class)}</code> <span class=arrow>→</span> <code>{esc(d.new_class)}</code>'
        elif d.kind == "added":
            head = f'<b class="chip add">+ estimator</b> <code>{esc(d.new_class)}</code>'
        elif d.kind == "removed":
            head = f'<b class="chip rem">− estimator</b> <code>{esc(d.old_class)}</code>'
        else:
            head = f'<b>{esc(d.new_class)} hyperparameters</b>'
        # Only a same-class hyperparameter change gets a per-param table. A swap
        # of estimator classes differs on ~every param (mostly None-vs-None), so
        # the headline alone is the signal.
        tbl = ""
        if d.kind == "params":
            rows = []
            for k, (o, n) in sorted(d.changed.items()):
                rows.append(f"<tr><td class=mono>{esc(k)}</td><td class=mono>{esc(o)}</td><td class=arrow>→</td><td class=mono>{esc(n)}</td></tr>")
            for k, n in sorted(d.added.items()):
                rows.append(f'<tr><td class=mono>{esc(k)}</td><td class=muted>—</td><td class=arrow>→</td><td class="mono add">{esc(n)}</td></tr>')
            for k, o in sorted(d.removed.items()):
                rows.append(f'<tr><td class=mono>{esc(k)}</td><td class=mono>{esc(o)}</td><td class=arrow>→</td><td class=muted>—</td></tr>')
            tbl = ("<table>" + "".join(rows) + "</table>") if rows else ""
        blocks.append(f"<div style='margin:8px 0'>{head}{tbl}</div>")
    return "<h3>Estimator changes</h3>" + "".join(blocks)


def _op_list(nodes, css):
    if not nodes:
        return '<span class="muted">none</span>'
    lis = "".join(f'<li class=mono>{esc(n.label)}</li>' for n in nodes)
    return f'<ul class="ops {css}">{lis}</ul>'


def _legend():
    items = [("frontier", "#b7f0c6", "new operation"),
             ("bubbled", "#fde6b0", "ancestor shifted by a change below it"),
             ("shared", "#eceff3", "unchanged"),
             ("choice", "#e7d4ff", "choice (ablation branch)"),
             ("estimator", "#ffffff", "estimator (blue border)")]
    return '<div class="legend">' + "".join(
        f'<span><span class="sw" style="background:{c}"></span>{esc(l)} — {esc(d)}</span>'
        for l, c, d in items) + "</div>"


def _pipeline_section(node, diff: DagDiff | None, lineage: Lineage,
                      index: int, show_dag: bool):
    p = node.pipeline
    anchor = f"pipe-{node.name}"
    parent_link = (f'<a href="#pipe-{esc(node.parent)}">{esc(node.parent)}</a>'
                   if node.parent and node.parent in lineage.nodes else esc(node.parent) or "—")
    score = f"{node.score:.5f}" if node.score is not None else "—"
    delta = lineage.delta_score(node.name)

    focus = (f'<a class="pill focus" href="#pa-explorer" data-focus="{index}" '
             f'title="tick only this pipeline in the explorer, coloured against '
             f'its parent">show in explorer ↗</a>')
    head = f"""
<h2 id="{anchor}">{esc(node.name)} {focus}</h2>
<div class="meta">
  <div><span class="score">{score}</span> {_delta_span(delta, lineage.improvement(node.name))} <span class="pill">{esc(node.phase or node.metric or 'score')}</span></div>
  <div class="muted">parent: {parent_link}</div>
  {f'<div class="muted">agent run: {node.duration_s:.0f}s</div>' if node.duration_s else ''}
  {f'<div class="muted">stratum: {_fmt_secs(node.runtime["wall_s"])}</div>' if node.runtime else ''}
</div>
<p class="sub">{esc(node.description or '')}</p>
"""
    if not p.ok:
        return head + f'<div class="card err">Could not extract DAG — {esc((p.error or "").splitlines()[0])}</div>'

    hist = f"<h3>Operator counts vs parent</h3>{_hist_delta_html(diff)}" if diff and diff.parent else \
           f"<h3>Operators</h3>{_hist_counts(p.dag)}"

    if diff and diff.parent:
        added = diff.added_nodes(frontier_only=True)
        removed = diff.removed_nodes()
        changes = f"""
<div class="grid2">
  <div><h3>Added operations ({len(added)})</h3>{_op_list(added, 'add')}</div>
  <div><h3>Removed operations ({len(removed)})</h3>{_op_list(removed, 'rem')}</div>
</div>
{_est_delta_html(diff)}
"""
    else:
        changes = _est_delta_html(diff) if diff else ""

    grid = ""
    if node.grid:
        grows = "".join(
            "<tr>" + "".join(f"<td class=mono>{esc(v)}</td>" for v in row.values()) + "</tr>"
            for row in node.grid)
        ghead = "".join(f"<th>{esc(k)}</th>" for k in node.grid[0].keys())
        grid = f"<h3>choose_from grid (from results.json)</h3><table><tr>{ghead}</tr>{grows}</table>"

    runtime = f'<div class="card">{_runtime_block(node.runtime)}</div>' if node.runtime else ""
    dag = ""
    if show_dag:
        status = diff_status_map(diff) if (diff and diff.parent) else None
        summary = (f"Operator DAG{' (diff vs parent)' if status else ''} · "
                   f"{len(p.dag.nodes)} ops")
        dag = f"""
<details class="dagbox">
<summary>{summary}</summary>
{_legend()}
<div class="canvas">{render_dag(p.dag, status)}</div>
</details>
"""
    return head + f"""
<div class="card">{hist}{changes}{grid}</div>{runtime}{dag}
"""


def _fmt_secs(seconds):
    if seconds is None:
        return "—"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.2f}s"


def _memory_svg(mem: dict, width: int = 620, height: int = 96) -> str:
    """RSS over the run as an inline area chart.

    The shaded band is the scored grid search; anything left of it is plan
    construction, so the ramp inside the band is the pipeline's own data loading
    and fitting. Drawn by hand rather than via graphviz because it is a plot, and
    it has to stay legible in both themes -- hence CSS variables for every colour.
    """
    curve = mem.get("curve") or []
    if len(curve) < 2:
        return ""
    pad_l, pad_r, pad_t, pad_b = 44, 8, 10, 16
    t_max = max(t for t, _ in curve) or 1.0
    m_max = max(m for _, m in curve) or 1.0

    def x(t):
        return pad_l + (width - pad_l - pad_r) * (t / t_max)

    def y(m):
        return pad_t + (height - pad_t - pad_b) * (1 - m / m_max)

    pts = " ".join(f"{x(t):.1f},{y(m):.1f}" for t, m in curve)
    area = (f"{x(curve[0][0]):.1f},{y(0):.1f} {pts} "
            f"{x(curve[-1][0]):.1f},{y(0):.1f}")

    band = ""
    if mem.get("scored_from_s") is not None and mem.get("scored_to_s") is not None:
        x0, x1 = x(min(mem["scored_from_s"], t_max)), x(min(mem["scored_to_s"], t_max))
        band = (f'<rect class="scored" x="{x0:.1f}" y="{pad_t}" '
                f'width="{max(x1 - x0, 1):.1f}" height="{height - pad_t - pad_b}"/>')

    peak_y = y(mem["peak_mb"]) if mem.get("peak_mb") else None
    peak = ("" if peak_y is None else
            f'<line class="peak" x1="{pad_l}" y1="{peak_y:.1f}" '
            f'x2="{width - pad_r}" y2="{peak_y:.1f}"/>')

    return f"""<svg class="memchart" viewBox="0 0 {width} {height}" role="img"
 aria-label="resident memory over the run">
{band}<polygon class="area" points="{area}"/><polyline class="line" points="{pts}"/>{peak}
<text x="0" y="{pad_t + 4:.0f}">{mem['peak_mb']:.0f} MB</text>
<text x="0" y="{height - pad_b:.0f}">0</text>
<text x="{pad_l}" y="{height - 4}">0s</text>
<text x="{width - pad_r}" y="{height - 4}" text-anchor="end">{t_max:.1f}s</text>
</svg>"""


def _runtime_block(rt: dict, top: int = 8):
    """Measured stratum runtime for one pipeline: totals plus its heavy hitters.

    ``wall_s`` is the whole scored grid search; ``op_time_s`` only the operator
    bodies, so the gap between them is scheduling, splitting and scoring
    overhead rather than lost time.
    """
    if not rt:
        return ""
    mem = rt.get("memory") or {}
    fields = [("wall", _fmt_secs(rt.get("wall_s"))),
              ("in operators", _fmt_secs(rt.get("op_time_s"))),
              ("buffer pool", _fmt_secs(rt.get("buffer_overhead_s"))),
              ("peak RSS", f"{rt['max_rss_mb']:.0f} MB" if rt.get("max_rss_mb") else "—"),
              ("op calls", rt.get("n_op_calls") or "—")]
    if mem.get("mean_mb") is not None:
        fields.insert(4, ("mean RSS", f"{mem['mean_mb']:.0f} MB"))
    pool = rt.get("pool") or {}
    if pool.get("hit_rate") is not None:
        fields.append(("pool hits", f"{100 * pool['hit_rate']:.0f}%"))
    if rt.get("best_score") is not None:
        # This run's own score, which is not the agent's score above: it comes
        # from stratum's scheduler and, on a sampled sweep, from fewer rows.
        n = len(rt.get("scores") or ())
        label = "best of %d measured" % n if n > 1 else "score measured"
        fields.append((label, f"{rt['best_score']:.5f}"))
    strip = "".join(f'<span><span class="k">{esc(k)}</span> <b>{esc(v)}</b></span>'
                    for k, v in fields)

    ops = rt.get("ops") or []
    total = sum(r["time_s"] for r in ops) or 1.0
    rows = "".join(
        f'<tr><td class=mono>{esc(r["op"])}</td><td class=num>{r["count"]}</td>'
        f'<td class=num>{r["time_s"]:.3f}</td>'
        f'<td class=num>{100 * r["time_s"] / total:.1f}%</td></tr>'
        for r in ops[:top])
    rest = len(ops) - top
    if rest > 0:
        other = sum(r["time_s"] for r in ops[top:])
        rows += (f'<tr><td class="muted">{rest} more operator type(s)</td>'
                 f'<td class=num></td><td class=num>{other:.3f}</td>'
                 f'<td class=num>{100 * other / total:.1f}%</td></tr>')
    table = (f"<table><tr><th>operator</th><th class=num>calls</th>"
             f"<th class=num>time (s)</th><th class=num>share</th></tr>{rows}</table>")

    note = []
    if rt.get("sample_rows"):
        note.append(f"measured on a {rt['sample_rows']:,}-row sample")
    if rt.get("cv"):
        note.append(f"cv: {rt['cv']}")
    sub = (f'<p class="muted" style="margin:6px 0 0">{esc(" · ".join(note))}</p>'
           if note else "")
    chart = _memory_svg(mem)
    if chart:
        chart = (f"<h3>Resident memory</h3>{chart}"
                 f'<p class="muted" style="margin:0">Sampled every '
                 f"{mem['interval_s']}s over {mem['duration_s']:.1f}s "
                 f"({mem['n_samples']} samples); shaded band is the scored grid "
                 f"search, peak {mem['peak_mb']:.0f} MB at "
                 f"{mem['peak_at_s']:.1f}s.</p>")
    return (f'<h3>Measured runtime (stratum)</h3><div class="rt">{strip}</div>'
            f"{table}{sub}{chart}")


def _runtime_section(lineage: Lineage):
    """One table ranking every measured pipeline by wall time."""
    measured = [n for n in lineage.ordered() if n.runtime]
    if not measured:
        return ""
    measured.sort(key=lambda n: -(n.runtime.get("wall_s") or 0))
    rows = "".join(
        "<tr>"
        f'<td><a href="#pipe-{esc(n.name)}">{esc(n.name)}</a></td>'
        f'<td>{esc(n.phase or "")}</td>'
        f'<td class=num>{_fmt_secs(n.runtime.get("wall_s"))}</td>'
        f'<td class=num>{_fmt_secs(n.runtime.get("op_time_s"))}</td>'
        f'<td class=num>{n.runtime.get("max_rss_mb") or 0:.0f}</td>'
        f'<td class=num>{(n.runtime.get("memory") or {}).get("mean_mb") or 0:.0f}</td>'
        f'<td class=num>{n.runtime.get("n_op_calls") or 0}</td>'
        f'<td class=num>{"—" if n.score is None else f"{n.score:.5f}"}</td>'
        "</tr>"
        for n in measured)
    samples = {n.runtime.get("sample_rows") for n in measured}
    note = ""
    if samples - {None}:
        shown = ", ".join(f"{s:,}" if s else "full data" for s in sorted(
            samples, key=lambda s: (s is None, s)))
        note = (f'<p class="muted">Rows read per pipeline: {shown}. '
                f"Wall time is the scored grid search only, excluding process "
                f"start-up and plan construction.</p>")
    total = sum(n.runtime.get("wall_s") or 0 for n in measured)
    return f"""
<h2 id="agg-runtime">Measured runtime — {len(measured)} pipeline(s), {_fmt_secs(total)} total</h2>
{note}
<div class="card"><table>
<tr><th>pipeline</th><th>phase</th><th class=num>wall</th><th class=num>operators</th>
<th class=num>peak MB</th><th class=num>mean MB</th><th class=num>op calls</th>
<th class=num>score</th></tr>
{rows}
</table></div>
"""


def _hist_counts(dag):
    rows = "".join(f"<tr><td class=mono>{esc(t)}</td><td>{c}</td></tr>"
                   for t, c in sorted(dag.histogram().items()))
    return f"<table><tr><th>operator</th><th>count</th></tr>{rows}</table>"


def _stats_block(heading, anchor, per_pipe, *, note=""):
    """One operator-statistics table over a list of per-pipeline histograms.

    For each operator type: absolute total (summed over all pipelines), how many
    pipelines contain it, and the per-pipeline count distribution
    (mean / median / std / min / max). Distribution stats span *all* pipelines in
    ``per_pipe``, so a pipeline lacking the op contributes 0.
    """
    N = len(per_pipe)
    if N == 0:
        return ""
    op_types = set().union(*per_pipe)

    stats = []
    for t in op_types:
        counts = [h.get(t, 0) for h in per_pipe]
        stats.append((
            t,
            sum(counts),                                   # absolute total
            sum(1 for c in counts if c > 0),               # pipelines present
            statistics.mean(counts),
            statistics.median(counts),
            statistics.pstdev(counts) if N > 1 else 0.0,
            min(counts),
            max(counts),
        ))
    stats.sort(key=lambda r: (-r[1], r[0]))

    rows = "".join(
        f"<tr><td class=mono>{esc(t)}</td>"
        f"<td class=num>{total}</td><td class=num>{present}/{N}</td>"
        f"<td class=num>{mean:.2f}</td><td class=num>{median:g}</td>"
        f"<td class=num>{std:.2f}</td><td class=num>{mn}</td><td class=num>{mx}</td></tr>"
        for (t, total, present, mean, median, std, mn, mx) in stats)
    table = (
        "<table><tr><th>operator</th><th class=num>total</th><th class=num>present</th>"
        "<th class=num>mean</th><th class=num>median</th><th class=num>std</th>"
        f"<th class=num>min</th><th class=num>max</th></tr>{rows}</table>")

    sizes = [sum(h.values()) for h in per_pipe]
    summary = (
        f'<p class="muted">{N} pipelines · {len(op_types)} distinct operator types · '
        f'{sum(sizes)} operator instances total · DAG size per pipeline: '
        f'median {statistics.median(sizes):g}, min {min(sizes)}, max {max(sizes)}.{note} '
        f'Distribution stats span all {N} pipelines (a pipeline lacking an op counts as 0).</p>')
    return f'<h2 id="{anchor}" style="border:0">{esc(heading)}</h2>{summary}<div class="card">{table}</div>'


def _aggregate_section(lineage: Lineage):
    """High-level operator statistics: one table at the logical-IR altitude, and
    one at the physical altitude (default lowering + implementation selection)."""
    ok = [n for n in lineage.ordered() if n.pipeline.ok]
    if not ok:
        return ""

    out = _stats_block("Operator statistics — logical IR", "agg-logical",
                       [n.pipeline.dag.histogram() for n in ok])

    phys_ok = [n for n in ok if n.pipeline.phys_dag is not None]
    if phys_ok:
        missing = len(ok) - len(phys_ok)
        note = (" Physical lowering + default implementation selection; ops are"
                " split by operation kind (e.g. NumericOp[square],"
                " PandasColumnSelectorOp[glob]), and assign-maps by their full"
                " symbolic expression.")
        if missing:
            note += f" {missing} pipeline(s) omitted (physical extraction failed)."
        out += _stats_block("Operator statistics — physical (default selection)",
                            "agg-physical",
                            [n.pipeline.phys_dag.histogram(specific=True) for n in phys_ok],
                            note=note)
    return out


def _tree_section(lineage: Lineage, merged):
    """The search tree, in one or two pre-rendered colourings.

    Both are rendered server-side (graphviz lays out trees better than anything
    worth hand-writing) and toggled client-side, so switching colouring costs
    nothing at view time. Clicking a node ticks that pipeline in the explorer.
    """
    trees = [("delta", "score vs parent", render_lineage(lineage, color_by="delta"))]
    if merged.phase_colors:
        trees.append(("phase", "search phase",
                      render_lineage(lineage, color_by="phase",
                                     phase_colors=merged.phase_colors)))
    modes = ""
    if len(trees) > 1:
        radios = "".join(
            f'<label><input type="radio" name="pa-tree" value="{key}"'
            f'{" checked" if i == 0 else ""}>{esc(label)}</label>'
            for i, (key, label, _) in enumerate(trees))
        phases = "".join(
            f'<span><span class="dot" style="background:{c}"></span>{esc(ph)}</span>'
            for ph, c in merged.phase_colors.items())
        modes = (f'<div class="pa-treemodes">colour by: {radios}</div>'
                 f'<div class="legend">{phases}</div>')
    panes = "".join(
        f'<div class="canvas" data-tree="{key}"{"" if i == 0 else " hidden"}>{svg}</div>'
        for i, (key, _, svg) in enumerate(trees))
    lower = " (lower is better)" if lineage.lower_is_better else ""
    return f"""
<h2 id="tree" style="border:0">Search tree</h2>
<p class="muted">Every step the agent ran, linked to the step it was derived from.
Fill: green improved on the parent&#160;· amber flat&#160;· red regressed{lower}.
<b>Click a node</b> to tick that pipeline in the explorer below (a thick outline
marks the ticked ones); shift-click takes its whole subtree.</p>
{modes}
{panes}
"""


def _explorer_section(merged):
    """Picker + merged DAG + node inspector. All behaviour lives in
    ``explorer.js``; this is the markup it binds to."""
    share = merged.share_histogram()
    n_nodes = len(merged.nodes)
    n_pipes = sum(1 for p in merged.pipelines if p["ok"])
    unique = share.get(1, 0)
    return f"""
<h2 id="explorer" style="border:0">Operator explorer</h2>
<p class="muted">One graph for the whole run: the {n_pipes} analyzed pipelines
hold {sum(p["ops"] for p in merged.pipelines)} operations, which merge into
<b>{n_nodes}</b> distinct ones &#8212; an operation two pipelines share is one
node here, keyed by the content signature of its whole sub-computation, so it is
the same node only if it really is the same computation. {unique} of them occur in
a single pipeline. Tick pipelines to overlay them.</p>
<div class="pa-explorer" id="pa-explorer">
  <div class="pa-side">
    <h3>Pipelines &#160;<span id="pa-count"></span></h3>
    <div class="pa-acts">
      <button id="pa-all">all</button>
      <button id="pa-none">none</button>
      <button id="pa-invert">invert</button>
      <button id="pa-best" title="root &#8594; best-scoring pipeline">best path</button>
      <button id="pa-roots">roots</button>
    </div>
    <input id="pa-filter" placeholder="filter by name&#8230;">
    <div id="pa-list"></div>
  </div>
  <div class="pa-main">
    <div class="pa-tools">
      <label>colour: <select id="pa-mode">
        <option value="share">how widely shared</option>
        <option value="pipe">by pipeline</option>
        <option value="diff">diff vs parent</option>
      </select></label>
      <button id="pa-zin" title="zoom in">+</button>
      <button id="pa-zout" title="zoom out">&#8722;</button>
      <button id="pa-fit">reset view</button>
      <button id="pa-full" title="give the graph the whole window (Esc to leave)">full screen</button>
      <span class="muted">scroll to zoom, drag to pan, click an operation for its
      pipelines</span>
      <span id="pa-modehint"></span>
    </div>
    <div class="legend" id="pa-legend"></div>
    <div class="pa-canvas"><svg id="pa-svg"></svg></div>
    <p id="pa-stats"></p>
    <div class="card" id="pa-inspect"></div>
  </div>
</div>
"""


def _payload(merged) -> str:
    """The explorer's data, inlined as JSON. ``</`` is escaped so no label can
    close the script element early."""
    return merged.to_json().replace("</", "<\\/")


def build_html(lineage: Lineage, *, title="Pipeline evolution", subtitle="",
               generated_note="", per_pipeline_dags=False):
    ordered = lineage.ordered()
    ok = sum(1 for n in ordered if n.pipeline.ok)
    merged = build_merged(lineage)

    sections = []
    for i, node in enumerate(ordered):
        parent = lineage.nodes.get(node.parent) if node.parent else None
        diff = None
        if node.pipeline.ok:
            parent_dag = parent.pipeline.dag if (parent and parent.pipeline.ok) else None
            diff = diff_dags(parent_dag, node.pipeline.dag)
        sections.append(_pipeline_section(node, diff, lineage, i, per_pipeline_dags))

    nav = " · ".join([
        '<a href="#tree">search tree</a>',
        '<a href="#explorer">operator explorer</a>',
        '<a href="#agg-logical">operator statistics</a>',
        *(['<a href="#agg-runtime">measured runtime</a>']
          if any(n.runtime for n in ordered) else []),
        f'<a href="#{esc(ordered[0].name) if ordered else ""}">per-pipeline detail</a>',
    ])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{_CSS}
{_asset("explorer.css")}</style>
<div class="wrap">
<h1>{esc(title)}</h1>
<p class="sub">{esc(subtitle)} · {ok}/{len(ordered)} pipelines analyzed · logical IR · {esc(generated_note)} {ts}</p>
<p class="sub">{nav}</p>

{_tree_section(lineage, merged)}

{_explorer_section(merged)}

{_runtime_section(lineage)}

{_aggregate_section(lineage)}

<h2 style="border:0;margin-top:26px">Per-pipeline detail</h2>
<p class="muted">What each step changed against its parent, in counts, operations
and estimator hyperparameters. Use “show in explorer” on any of them to see the
same diff on the graph.</p>

{''.join(sections)}
</div>
<script type="application/json" id="pa-data">{_payload(merged)}</script>
<script>{_asset("explorer.js")}</script>
"""
