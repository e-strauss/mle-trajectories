"""pipeline_analyzer — analyze how skrub DataOps pipelines evolve across iterations.

Loads each pipeline file, extracts its stratum *logical* operator DAG (no data
needed), and diffs each pipeline against its PARENT to show what stayed the same
and what changed — structurally (added/removed operations) and in estimator
hyperparameters. Emits a self-contained HTML report with a lineage tree and a
diff-colored operator DAG per pipeline.

Depends on ``stratum`` (imported as ``stratum.optimizer``), so it works both in
this repo and later as an installed dependency.
"""
from .loader import Pipeline, load_pipeline, load_all, discover, resolve
from .dag import Dag, Node, build_dag
from .diff import DagDiff, diff_dags, is_structural_noop
from .lineage import (Lineage, build_lineage, build_lineage_from_trajectory,
                      fold_translation_variants)
from .trajectory import (PARSERS, Step, Trajectory, detect_type, parse,
                         parse_mle_star, parse_mlevolve)
from .html import build_html

__all__ = [
    "Pipeline", "load_pipeline", "load_all", "discover", "resolve",
    "Dag", "Node", "build_dag",
    "DagDiff", "diff_dags", "is_structural_noop",
    "Lineage", "build_lineage", "build_lineage_from_trajectory",
    "fold_translation_variants",
    "PARSERS", "Step", "Trajectory", "detect_type", "parse",
    "parse_mle_star", "parse_mlevolve",
    "build_html",
]
