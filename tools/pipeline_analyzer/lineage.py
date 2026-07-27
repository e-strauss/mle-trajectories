"""Assemble the pipeline lineage: a PARENT-linked tree annotated with scores.

Scores/metric/description come from ``results.json`` (written by the ml-score
harness) when present; PARENT comes from the pipeline files themselves. The two
can disagree slightly (a file's PARENT vs results.json's parent) -- the file wins
for the graph edges, results.json supplies the score.
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

    @property
    def description(self):
        return self.pipeline.description


@dataclass
class Lineage:
    nodes: dict                        # name -> LineageNode
    roots: list                        # names with no parent

    def ordered(self):
        """Depth-first from each root, so children follow parents."""
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
