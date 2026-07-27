"""Structural diff between two logical DAGs (parent -> child).

Because nodes are keyed by a recursive content signature, a set difference over
signatures already tells us what sub-computations are shared, added, or removed.
On top of that we compute two things that make the result *readable*:

* the **change frontier** -- the added nodes whose every input already existed in
  the parent. These are the operations genuinely introduced by this step; every
  other "added" node is just an ancestor whose signature shifted because a
  descendant under it changed (Merkle diffs bubble upward). Separating the two is
  the difference between "you added a soil-descriptor block" and "...and therefore
  the assign/predictor above it also count as new".
* **estimator deltas** -- estimator swaps and hyperparameter changes, aligned by
  logical family so an LGBM 200->700 tree bump reads as a param delta, not an
  opaque predictor swap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .dag import Dag, Node


@dataclass
class EstimatorDelta:
    kind: str                       # "swap" | "params" | "added" | "removed"
    old_class: str | None
    new_class: str | None
    changed: dict = field(default_factory=dict)   # param -> (old, new)
    added: dict = field(default_factory=dict)      # param -> new
    removed: dict = field(default_factory=dict)    # param -> old


@dataclass
class DagDiff:
    parent: Dag | None
    child: Dag
    shared: set                     # signatures in both
    added: set                      # signatures only in child
    removed: set                    # signatures only in parent
    frontier: set                   # added nodes whose inputs are all in parent
    hist_delta: dict                # op_type -> (parent_count, child_count)
    estimator_deltas: list          # list[EstimatorDelta]

    def added_nodes(self, frontier_only=False):
        sigs = self.frontier if frontier_only else self.added
        return [self.child.nodes[s] for s in self.child.order if s in sigs]

    def removed_nodes(self):
        if self.parent is None:
            return []
        return [self.parent.nodes[s] for s in self.parent.order if s in self.removed]


def diff_dags(parent: Dag | None, child: Dag) -> DagDiff:
    if parent is None:
        return DagDiff(
            parent=None, child=child,
            shared=set(), added=set(child.nodes), removed=set(),
            frontier={s for s, n in child.nodes.items() if not n.inputs},
            hist_delta={t: (0, c) for t, c in child.histogram().items()},
            estimator_deltas=[EstimatorDelta("added", None, n.estimator[0], added=n.estimator[1])
                              for n in child.estimators()],
        )

    p, c = set(parent.nodes), set(child.nodes)
    shared, added, removed = p & c, c - p, p - c

    # Frontier: an added node all of whose inputs already exist in the parent.
    frontier = {s for s in added if all(i in p for i in child.nodes[s].inputs)}

    # Histogram delta across all op types present in either DAG.
    ph, ch = parent.histogram(), child.histogram()
    hist_delta = {t: (ph.get(t, 0), ch.get(t, 0)) for t in sorted(set(ph) | set(ch))}

    return DagDiff(
        parent=parent, child=child,
        shared=shared, added=added, removed=removed, frontier=frontier,
        hist_delta=hist_delta,
        estimator_deltas=_estimator_deltas(parent, child),
    )


def _estimator_deltas(parent: Dag, child: Dag) -> list[EstimatorDelta]:
    """Align estimator ops by logical family and report swaps / param changes.

    Multiple estimators of the same family (e.g. the 3 branches of a choose_from
    ablation) are aligned by order within the family.
    """
    def by_family(dag: Dag):
        groups: dict[str, list[Node]] = {}
        for n in dag.estimators():
            groups.setdefault(n.family, []).append(n)
        return groups

    pg, cg = by_family(parent), by_family(child)
    deltas: list[EstimatorDelta] = []
    for fam in sorted(set(pg) | set(cg)):
        pl, cl = pg.get(fam, []), cg.get(fam, [])
        for i in range(max(len(pl), len(cl))):
            po = pl[i] if i < len(pl) else None
            co = cl[i] if i < len(cl) else None
            if po is None:
                deltas.append(EstimatorDelta("added", None, co.estimator[0], added=co.estimator[1]))
                continue
            if co is None:
                deltas.append(EstimatorDelta("removed", po.estimator[0], None, removed=po.estimator[1]))
                continue
            pc, ppar = po.estimator
            cc, cpar = co.estimator
            if pc != cc:
                deltas.append(EstimatorDelta("swap", pc, cc,
                                             changed={k: (ppar.get(k), cpar.get(k))
                                                      for k in set(ppar) | set(cpar)
                                                      if ppar.get(k) != cpar.get(k)}))
                continue
            changed = {k: (ppar[k], cpar[k]) for k in set(ppar) & set(cpar) if ppar[k] != cpar[k]}
            added = {k: cpar[k] for k in set(cpar) - set(ppar)}
            removed = {k: ppar[k] for k in set(ppar) - set(cpar)}
            if changed or added or removed:
                deltas.append(EstimatorDelta("params", pc, cc,
                                             changed=changed, added=added, removed=removed))
    return deltas


def is_structural_noop(diff: DagDiff) -> bool:
    """True when nothing about the operator graph changed except, possibly,
    estimator hyperparameters (the whole GBDT-capacity track)."""
    return not diff.frontier and not diff.removed
