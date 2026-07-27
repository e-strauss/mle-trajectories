"""Load a skrub DataOps pipeline file and extract its stratum logical DAG.

A pipeline file only *defines* a plan (module-level ``pred``, ``DESCRIPTION``,
optional ``PARENT`` -- see ``skrub_dataops_guide.md``). We import it to obtain
``pred`` and run stratum's ``logical_optimize`` to get the operator DAG.

No data is needed: importing a plan normally triggers eager preview computation
(which would read the workspace CSV), so we import *and* optimize inside
``skrub.config_context(eager_data_ops=False)``. That means the tool runs with no
input dataset present and never touches the 4M-row table.
"""
from __future__ import annotations

import importlib
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import skrub

from stratum.optimizer._optimize import logical_optimize, OptConfig

from .dag import Dag, build_dag

# Modules a pipeline file pulls from the workspace; must be re-imported fresh per
# pipeline so one file's plan never leaks into the next.
_WORKSPACE_MODULES = ("common", "features", "nn")


@dataclass
class Pipeline:
    name: str                     # e.g. "pipeline_05"
    path: Path
    parent: str | None
    description: str | None
    doc: str | None               # module docstring (the agent's rationale)
    dag: Dag | None               # None if extraction failed
    error: str | None = None      # populated when import/extraction failed

    @property
    def ok(self) -> bool:
        return self.dag is not None


def _fresh_import(module_name: str, pipe_dir: Path):
    if str(pipe_dir) not in sys.path:
        sys.path.insert(0, str(pipe_dir))
    for m in (module_name, *_WORKSPACE_MODULES):
        sys.modules.pop(m, None)
    return importlib.import_module(module_name)


def load_pipeline(module_name: str, pipe_dir: Path, *, unroll_choices: bool = False) -> Pipeline:
    """Import ``module_name`` from ``pipe_dir`` and extract its logical DAG.

    Never raises: an import or extraction failure (e.g. a missing dependency) is
    captured on the returned :class:`Pipeline` as ``error`` with ``dag=None``.
    """
    path = pipe_dir / f"{module_name}.py"
    try:
        with skrub.config_context(eager_data_ops=False):
            mod = _fresh_import(module_name, pipe_dir)
            pred = getattr(mod, "pred", None)
            if pred is None:
                raise AttributeError(f"{module_name} defines no module-level `pred`")
            root = logical_optimize(pred, OptConfig(unroll_choices=unroll_choices))
            dag = build_dag(root)
        return Pipeline(
            name=module_name, path=path,
            parent=getattr(mod, "PARENT", None),
            description=getattr(mod, "DESCRIPTION", None),
            doc=(mod.__doc__ or "").strip() or None,
            dag=dag,
        )
    except Exception as e:  # noqa: BLE001 - want to report, not crash the batch
        return Pipeline(
            name=module_name, path=path,
            parent=_safe_module_attr(module_name, pipe_dir, "PARENT"),
            description=_safe_module_attr(module_name, pipe_dir, "DESCRIPTION"),
            doc=None, dag=None,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )


def _safe_module_attr(module_name, pipe_dir, attr):
    """Best-effort read of a top-level ``PARENT``/``DESCRIPTION`` assignment via a
    text scan, so a pipeline that fails to import still slots into the lineage."""
    try:
        text = (pipe_dir / f"{module_name}.py").read_text()
    except OSError:
        return None
    import re
    m = re.search(rf"^{attr}\s*=\s*(.+)$", text, re.MULTILINE)
    if not m:
        return None
    val = m.group(1).strip()
    if val in ("None", "null"):
        return None
    return val.strip("\"'")


def discover(pipe_dir: Path) -> list[str]:
    """Sorted ``pipeline_*`` module names present in ``pipe_dir``."""
    return sorted(p.stem for p in pipe_dir.glob("pipeline_*.py"))


def load_all(pipe_dir: Path, *, unroll_choices: bool = False) -> list[Pipeline]:
    return [load_pipeline(name, pipe_dir, unroll_choices=unroll_choices)
            for name in discover(pipe_dir)]
