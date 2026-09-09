"""pipeline_analyzer — analyze how skrub DataOps pipelines evolve across iterations.

Loads each pipeline file, extracts its stratum *logical* operator DAG (no data
needed), and diffs each pipeline against its PARENT to show what stayed the same
and what changed — structurally (added/removed operations) and in estimator
hyperparameters. Emits a self-contained HTML report: the search tree, an
interactive explorer over the *merged* operator DAG of the whole run (every
pipeline overlaid, shared operations collapsed onto one node), and a per-pipeline
account of what each step changed.

Depends on ``stratum`` (imported as ``stratum.optimizer``), so it works both in
this repo and later as an installed dependency.
"""
from .loader import Pipeline, load_pipeline, load_all, discover, resolve
from .dag import Dag, Node, build_dag
from .diff import DagDiff, diff_dags, is_structural_noop
from .lineage import (Lineage, build_lineage, build_lineage_from_trajectory,
                      fold_translation_variants)
from .merged import MergedDag, MergedNode, build_merged
from .trajectory import (PARSERS, Step, Trajectory, detect_type, parse,
                         parse_mle_star, parse_mlevolve)
from .html import build_html

__all__ = [
    "Pipeline", "load_pipeline", "load_all", "discover", "resolve",
    "Dag", "Node", "build_dag",
    "DagDiff", "diff_dags", "is_structural_noop",
    "Lineage", "build_lineage", "build_lineage_from_trajectory",
    "fold_translation_variants",
    "MergedDag", "MergedNode", "build_merged",
    "PARSERS", "Step", "Trajectory", "detect_type", "parse",
    "parse_mle_star", "parse_mlevolve",
    "build_html",
    "default_store_path", "load_store",
]


def __getattr__(name):
    """Expose the runtime-store helpers without importing the module eagerly.

    ``runtime`` is also run as ``python -m pipeline_analyzer.runtime``; importing
    it here would make runpy warn that the module was already in sys.modules.
    """
    if name in ("default_store_path", "load_store"):
        from . import runtime
        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
