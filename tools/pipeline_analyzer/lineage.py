"""Assemble the pipeline lineage: a parent-linked tree annotated with scores.

Two sources of edges and scores, one tree shape:

* :func:`build_lineage` -- the pipeline files themselves declare ``PARENT`` and
  ``DESCRIPTION``, with scores from ``results.json`` (written by the ml-score
  harness). The two can disagree slightly (a file's PARENT vs results.json's
  parent) -- the file wins for the graph edges, results.json supplies the score.
* :func:`build_lineage_from_trajectory` -- an agent run's trajectory supplies
  parents, scores, timings and rationale, and the pipeline files carry no
  annotations at all. This is the path for hand-skrubified runs, where the
  skrubified folder holds a plain rewrite of each step's code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .loader import Pipeline


@dataclass
class LineageNode:
    name: str
    pipeline: Pipeline
    parent: str | None
    children: list = field(default_factory=list)
    score: float | None = None
    metric: str | None = None
    duration_s: float | None = None
    grid: list | None = None          # extra.grid rows for choose_from runs
    desc: str | None = None           # rationale, when it comes from outside the file
    phase: str | None = None          # trajectory phase (Init / Ablation / Improve / ...)
    order: tuple | None = None        # chronological sort key, when known
    returncode: int | None = None

    @property
    def description(self):
        return self.desc or self.pipeline.description


@dataclass
class Lineage:
    nodes: dict                        # name -> LineageNode
    roots: list                        # names with no parent

    def ordered(self):
        """Report order: chronological when every node knows when it ran (a
        trajectory-built lineage -- a parent always ran before its children, so
        this keeps parents first while reading as the run happened), otherwise
        depth-first from each root."""
        if self.nodes and all(n.order is not None for n in self.nodes.values()):
            return sorted(self.nodes.values(), key=lambda n: n.order)
        seen, out = set(), []

        def walk(name):
            if name in seen:
                return
            seen.add(name)
            out.append(self.nodes[name])
            for ch in self.nodes[name].children:
                walk(ch)

        for r in self.roots:
            walk(r)
        # Any node not reachable from a root (dangling parent) still gets emitted.
        for name in self.nodes:
            if name not in seen:
                walk(name)
        return out

    def delta_score(self, name):
        node = self.nodes[name]
        if node.score is None or node.parent is None:
            return None
        parent = self.nodes.get(node.parent)
        if parent is None or parent.score is None:
            return None
        return node.score - parent.score


def load_results(results_path: Path) -> dict:
    if not results_path.exists():
        return {}
    try:
        rows = json.loads(results_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {r["pipeline"]: r for r in rows if "pipeline" in r}


def build_lineage(pipelines: list[Pipeline], results_path: Path | None = None) -> Lineage:
    results = load_results(results_path) if results_path else {}
    nodes: dict[str, LineageNode] = {}
    for p in pipelines:
        r = results.get(p.name, {})
        nodes[p.name] = LineageNode(
            name=p.name, pipeline=p, parent=p.parent,
            score=r.get("score"), metric=r.get("metric"),
            duration_s=r.get("duration_s"),
            grid=(r.get("extra") or {}).get("grid"),
        )
    roots = []
    for name, node in nodes.items():
        if node.parent and node.parent in nodes:
            nodes[node.parent].children.append(name)
        else:
            roots.append(name)
    for node in nodes.values():
        node.children.sort()
    return Lineage(nodes=nodes, roots=sorted(roots))


def fold_translation_variants(pipelines: list[Pipeline], traj) -> list[tuple[str, str]]:
    """Make steps whose *original* code is byte-identical share one DAG.

    When each step is skrubified independently, two rewrites of the same original
    differ in ways the original does not -- a row filter written as ``isin`` in one
    file and ``map(...) >= n`` in the next re-signatures every node above it, so a
    step that changed *nothing* reads as a near-total rewrite. (On run4 that noise
    floor is ~14% shared nodes between rewrites of identical code.)

    The trajectory knows which steps are code-identical, so the fix is to let the
    first such step's DAG stand for the group: any remaining difference is
    translation noise by construction. Mutates ``pipelines`` in place and returns
    the ``(module, representative)`` pairs it folded.
    """
    steps = traj.by_module()
    by_name = {p.name: p for p in pipelines}
    rep: dict[str, str] = {}
    folded = []
    for st in sorted(traj.steps, key=lambda s: s.order):
        p = by_name.get(st.module)
        if p is None or st.code_sig is None or not p.ok:
            continue
        first = rep.setdefault(st.code_sig, st.module)
        if first != st.module:
            src = by_name[first]
            p.dag, p.phys_dag, p.phys_error = src.dag, src.phys_dag, src.phys_error
            folded.append((st.module, first))
    return folded


def build_lineage_from_trajectory(pipelines: list[Pipeline], traj) -> Lineage:
    """Lineage for a run whose pipeline files carry no ``PARENT``: the trajectory
    (see ``pipeline_analyzer.trajectory``) supplies the edges and the scores.

    Only steps that were actually skrubified become nodes. A parent that was not
    skrubified is replaced by its nearest skrubified ancestor, so a partial folder
    still yields one connected tree instead of a pile of roots -- the diff then
    spans the skipped steps, which is the honest reading of it.
    """
    steps = traj.by_module()
    by_name = {p.name: p for p in pipelines}

    def nearest_present(module: str | None) -> str | None:
        seen = set()
        while module and module not in by_name:
            if module in seen:                 # defensive: never loop on a cycle
                return None
            seen.add(module)
            step = steps.get(module)
            module = step.parent if step else None
        return module

    nodes: dict[str, LineageNode] = {}
    for name, pipe in by_name.items():
        st = steps.get(name)
        nodes[name] = LineageNode(
            name=name, pipeline=pipe,
            parent=nearest_present(st.parent) if st else None,
            score=st.score if st else None,
            duration_s=st.time_s if st else None,
            desc=st.desc if st else None,
            phase=st.phase if st else None,
            order=st.order if st else None,
            returncode=st.returncode if st else None,
        )
    roots = []
    for name, node in nodes.items():
        if node.parent == name or node.parent not in nodes:
            node.parent = None
        if node.parent:
            nodes[node.parent].children.append(name)
        else:
            roots.append(name)
    key = lambda n: (nodes[n].order or (), n)   # noqa: E731 - chronological
    for node in nodes.values():
        node.children.sort(key=key)
    return Lineage(nodes=nodes, roots=sorted(roots, key=key))
