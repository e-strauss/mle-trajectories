"""Graphviz rendering: operator DAGs (with diff coloring) and the lineage tree.

Everything returns an inline ``<svg>`` string (prelude stripped) so it embeds
directly into the self-contained HTML report. SVGs are shown on a light card in
the page, so node fills are chosen to read on a light canvas in either page theme.
"""
from __future__ import annotations

from graphviz import Digraph

from .dag import Dag
from .diff import DagDiff
from .lineage import Lineage

# Diff status -> (fill, border)
_STATUS_STYLE = {
    "shared":   ("#eceff3", "#9aa4b2"),
    "frontier": ("#b7f0c6", "#15a34a"),   # genuinely new operation
    "bubbled":  ("#fde6b0", "#d09a1e"),   # ancestor re-signatured by a descendant
    "choice":   ("#e7d4ff", "#8b5cf6"),
    "removed":  ("#f7c9c9", "#dc2626"),
}
_EST_BORDER = "#2563eb"


def _svg(dot: Digraph) -> str:
    raw = dot.pipe(format="svg").decode("utf-8")
    i = raw.find("<svg")
    return raw[i:] if i != -1 else raw


def _wrap(text: str, width: int = 26) -> str:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\\n".join(lines)


def render_dag(dag: Dag, status: dict[str, str] | None = None) -> str:
    """Render ``dag`` top-down (source -> prediction). ``status`` maps a node
    signature to a diff status key; unmapped nodes render as ``shared``."""
    status = status or {}
    dot = Digraph(graph_attr={"rankdir": "TB", "bgcolor": "transparent",
                              "nodesep": "0.25", "ranksep": "0.4"},
                  node_attr={"shape": "box", "style": "filled,rounded",
                             "fontname": "Helvetica", "fontsize": "11",
                             "penwidth": "1.4", "margin": "0.11,0.06"},
                  edge_attr={"color": "#9aa4b2", "arrowsize": "0.7"})
    for sig, node in dag.nodes.items():
        st = status.get(sig)
        if st is None:
            st = "choice" if node.is_choice else "shared"
        fill, border = _STATUS_STYLE.get(st, _STATUS_STYLE["shared"])
        label = node.label
        if node.estimator is not None:
            label = f"{node.family}: {node.estimator[0]}"
            border = _EST_BORDER
        dot.node(sig, _wrap(label), fillcolor=fill, color=border)
    for sig, node in dag.nodes.items():
        for inp in node.inputs:
            if inp in dag.nodes:
                dot.edge(inp, sig)
    return _svg(dot)


def diff_status_map(diff: DagDiff) -> dict[str, str]:
    status = {}
    for sig in diff.child.nodes:
        if sig in diff.frontier:
            status[sig] = "frontier"
        elif sig in diff.added:
            status[sig] = "bubbled"
        elif diff.child.nodes[sig].is_choice:
            status[sig] = "choice"
        else:
            status[sig] = "shared"
    return status


# --- lineage tree ------------------------------------------------------------
def _score_fill(delta):
    if delta is None:
        return "#e2e8f0"
    if delta > 0.0005:
        return "#b7f0c6"     # improved
    if delta < -0.0005:
        return "#f7c9c9"     # regressed
    return "#fde6b0"         # flat


def render_lineage(lineage: Lineage, anchor_prefix: str = "pipe-") -> str:
    dot = Digraph(graph_attr={"rankdir": "TB", "bgcolor": "transparent",
                              "nodesep": "0.28", "ranksep": "0.55"},
                  node_attr={"shape": "box", "style": "filled,rounded",
                             "fontname": "Helvetica", "fontsize": "11",
                             "penwidth": "1.3", "margin": "0.14,0.08"},
                  edge_attr={"color": "#94a3b8", "arrowsize": "0.8"})
    for name, node in lineage.nodes.items():
        delta = lineage.delta_score(name)
        score = f"{node.score:.5f}" if node.score is not None else "—"
        dstr = ""
        if delta is not None:
            dstr = f"\\n({'+' if delta >= 0 else ''}{delta:.4f})"
        border = "#334155" if node.pipeline.ok else "#dc2626"
        label = f"{name.replace('pipeline_', 'p')}\\n{score}{dstr}"
        dot.node(name, label, fillcolor=_score_fill(delta), color=border,
                 href=f"#{anchor_prefix}{name}", tooltip=(node.description or name))
    for name, node in lineage.nodes.items():
        if node.parent and node.parent in lineage.nodes:
            dot.edge(node.parent, name)
    return _svg(dot)
