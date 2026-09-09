"""One DAG for the whole run: the union of every pipeline's operator graph.

Nodes are keyed by their recursive content signature (see :mod:`.dag`), so an
operation two pipelines *share* is literally the same key in both -- the union is
therefore a plain dict merge, and "which pipelines contain this operation" is the
set of pipelines whose DAG carried that key.

The result is serialized to a compact JSON payload that the report's client-side
explorer lays out and re-renders as pipelines are ticked on and off
(``explorer.js``). Two things make it compact enough to inline for a 68-pipeline
run: nodes are referenced by index into one topologically ordered list, and each
node stores only its membership list (the reverse index, pipeline -> nodes, is
rebuilt in the browser).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .lineage import Lineage

# Categorical, assigned to phases in first-seen order. Readable on the light
# canvas the graph is drawn on, and distinguishable from the diff palette.
_PHASE_COLORS = [
    "#2563eb", "#0d9488", "#c2410c", "#7c3aed", "#b45309",
    "#be185d", "#4d7c0f", "#0369a1", "#9333ea", "#a16207",
]
_NO_PHASE = "#64748b"


@dataclass
class MergedNode:
    """One operation in the union DAG."""
    idx: int
    sig: str
    label: str
    op_type: str
    family: str
    estimator: str | None
    inputs: list = field(default_factory=list)     # indices, always < idx
    members: list = field(default_factory=list)    # pipeline indices


@dataclass
class MergedDag:
    nodes: list                  # MergedNode, topologically ordered
    pipelines: list              # dicts describing each pipeline (see build)
    phase_colors: dict           # phase name -> hex
    by_sig: dict                 # sig -> index
    lower_is_better: bool = False

    def share_histogram(self) -> dict:
        h = {}
        for n in self.nodes:
            k = len(n.members)
            h[k] = h.get(k, 0) + 1
        return h

    def to_json(self) -> str:
        """Payload for ``explorer.js``. Keys are short because this is inlined."""
        return json.dumps({
            "pipelines": self.pipelines,
            "nodes": [{"l": n.label, "t": n.op_type, "f": n.family,
                       "e": n.estimator, "i": n.inputs, "m": n.members}
                      for n in self.nodes],
            "phaseColors": self.phase_colors,
            "lowerIsBetter": self.lower_is_better,
        }, separators=(",", ":"))


def build_merged(lineage: Lineage) -> MergedDag:
    """Union every successfully extracted pipeline DAG in ``lineage``.

    Pipeline order is the report's own order (chronological when known), and each
    pipeline's DAG is walked in its topological order, so appending unseen
    signatures yields a *globally* topological node list: a node's inputs are part
    of its signature, hence identical in every pipeline that has it, and were
    therefore appended before it.
    """
    ordered = lineage.ordered()
    index_of = {n.name: i for i, n in enumerate(ordered)}

    phase_colors: dict[str, str] = {}
    for node in ordered:
        if node.phase and node.phase not in phase_colors:
            phase_colors[node.phase] = _PHASE_COLORS[len(phase_colors) % len(_PHASE_COLORS)]

    nodes: list[MergedNode] = []
    by_sig: dict[str, int] = {}
    for pi, node in enumerate(ordered):
        p = node.pipeline
        if not p.ok:
            continue
        for sig in p.dag.order:
            n = p.dag.nodes[sig]
            idx = by_sig.get(sig)
            if idx is None:
                idx = len(nodes)
                by_sig[sig] = idx
                nodes.append(MergedNode(
                    idx=idx, sig=sig,
                    label=(f"{n.family}: {n.estimator[0]}" if n.estimator else n.label),
                    op_type=n.op_type, family=n.family,
                    estimator=n.estimator[0] if n.estimator else None,
                    # An input that is not in ``by_sig`` yet cannot happen (see
                    # the docstring); guard anyway rather than emit a bad edge.
                    inputs=[by_sig[i] for i in n.inputs if i in by_sig],
                ))
            nodes[idx].members.append(pi)

    pipelines = []
    for pi, node in enumerate(ordered):
        p = node.pipeline
        sigs = p.dag.order if p.ok else []
        pipelines.append({
            "n": node.name,
            "ph": node.phase or "",
            "c": phase_colors.get(node.phase or "", _NO_PHASE),
            "s": node.score,
            "d": lineage.delta_score(node.name),
            "up": lineage.improvement(node.name),
            "p": index_of.get(node.parent) if node.parent else None,
            "ok": bool(p.ok),
            "ops": len(sigs),
            "root": by_sig.get(p.dag.root_sig) if p.ok else None,
            "desc": (node.description or "")[:400] or None,
        })

    return MergedDag(nodes=nodes, pipelines=pipelines,
                     phase_colors=phase_colors, by_sig=by_sig,
                     lower_is_better=lineage.lower_is_better)
