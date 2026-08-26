"""Assemble the self-contained, theme-aware HTML report."""
from __future__ import annotations

import html as _html
import statistics
from datetime import datetime

from .diff import DagDiff, diff_dags
from .lineage import Lineage
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
details.dagbox{margin:8px 0}
details.dagbox>summary{list-style:none;cursor:pointer;user-select:none;display:flex;align-items:center;gap:6px;
 font-size:14px;margin:18px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
details.dagbox>summary::-webkit-details-marker{display:none}
details.dagbox>summary::before{content:"\\25B8";display:inline-block;transition:transform .15s ease;color:var(--muted)}
details.dagbox[open]>summary::before{transform:rotate(90deg)}
details.dagbox>summary:hover{color:var(--fg)}
"""

_THEME_JS = """
<script>document.querySelectorAll('svg a').forEach(function(a){
 a.addEventListener('click',function(e){var h=a.getAttribute('xlink:href')||a.getAttribute('href');
 if(h&&h[0]==='#'){e.preventDefault();var el=document.querySelector(h);if(el)el.scrollIntoView({behavior:'smooth'});}});});
</script>
"""


def esc(x):
    return _html.escape(str(x)) if x is not None else ""


def _delta_span(delta):
    if delta is None:
        return '<span class="delta flat">—</span>'
    cls = "up" if delta > 0.0005 else "down" if delta < -0.0005 else "flat"
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


def _pipeline_section(node, diff: DagDiff | None, lineage: Lineage):
    p = node.pipeline
    anchor = f"pipe-{node.name}"
    parent_link = (f'<a href="#pipe-{esc(node.parent)}">{esc(node.parent)}</a>'
                   if node.parent and node.parent in lineage.nodes else esc(node.parent) or "—")
    score = f"{node.score:.5f}" if node.score is not None else "—"
    delta = lineage.delta_score(node.name)

    head = f"""
<h2 id="{anchor}">{esc(node.name)}</h2>
<div class="meta">
  <div><span class="score">{score}</span> {_delta_span(delta)} <span class="pill">{esc(node.phase or node.metric or 'score')}</span></div>
  <div class="muted">parent: {parent_link}</div>
  {f'<div class="muted">{node.duration_s:.0f}s</div>' if node.duration_s else ''}
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

    status = diff_status_map(diff) if (diff and diff.parent) else None
    svg = render_dag(p.dag, status)
    summary = f"Operator DAG{' (diff vs parent)' if status else ''} · {len(p.dag.nodes)} ops"
    return head + f"""
<div class="card">{hist}{changes}{grid}</div>
<details class="dagbox">
<summary>{summary}</summary>
<div class="canvas">{svg}</div>
</details>
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


def build_html(lineage: Lineage, *, title="Pipeline evolution", subtitle="", generated_note=""):
    ordered = lineage.ordered()
    ok = sum(1 for n in ordered if n.pipeline.ok)
    sections = []
    for node in ordered:
        parent = lineage.nodes.get(node.parent) if node.parent else None
        diff = None
        if node.pipeline.ok:
            parent_dag = parent.pipeline.dag if (parent and parent.pipeline.ok) else None
            diff = diff_dags(parent_dag, node.pipeline.dag)
        sections.append(_pipeline_section(node, diff, lineage))

    lineage_svg = render_lineage(lineage)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{_CSS}</style>
<div class="wrap">
<h1>{esc(title)}</h1>
<p class="sub">{esc(subtitle)} · {ok}/{len(ordered)} pipelines analyzed · logical IR · {esc(generated_note)} {ts}</p>

{_aggregate_section(lineage)}

<h2 style="border:0">Lineage</h2>
<p class="muted">Node fill: green improved · amber flat · red regressed vs parent. Click a node to jump to it.</p>
<div class="canvas">{lineage_svg}</div>

<h2 style="border:0;margin-top:26px">Legend (per-pipeline DAG diff)</h2>
{_legend()}

{''.join(sections)}
</div>
{_THEME_JS}
"""
