"""Turn a stratum logical operator DAG into a diffable, serializable model.

The one non-obvious piece is the *content signature*. stratum's own
``Op.structure_key()`` keys inputs by ``id()`` -- correct for CSE within a single
DAG, but two separately-built pipeline DAGs never share object ids, so it can't
align nodes across pipelines. We instead compute a recursive Merkle signature:

    sig(op) = sha1( op_type + canonical(config fields) + tuple(sig(input)) )

Two nodes in two different pipelines get the *same* signature iff they denote the
same sub-computation (same op type, same config, same inputs all the way down).
The config portion is content-hashed (``_field_str``) rather than keyed by object
identity, so numpy/pandas constants and estimator hyperparameters compare by value
across pipelines -- an LGBM 200->700 tree bump re-signatures its predictor, and an
identical soil-weight Series re-imported per pipeline still matches. (stratum's own
``config_key`` is id-based for unhashable leaves, which is right for in-DAG CSE but
would make identical feature blocks look changed across pipelines.)
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from sklearn.base import BaseEstimator

from stratum.optimizer.ir._base import OperandRef
from stratum.optimizer.ir._ops import BaseEstimatorOp, ChoiceOp
from stratum.optimizer._op_utils import topological_iterator


def _np():
    import numpy
    return numpy


def _pd():
    import pandas
    return pandas


def _dataop_type():
    try:
        from skrub._data_ops._data_ops import DataOp
        return DataOp
    except Exception:
        return ()


_HEX_ID = re.compile(r"0x[0-9a-fA-F]+")


def _hb(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:16]


def _stable_repr(v) -> str:
    """repr with hex object-ids scrubbed, so lambdas/opaque objects don't churn
    a signature merely because they live at a different address each import."""
    return _HEX_ID.sub("0xX", repr(v))


def _field_str(v) -> str:
    """Deterministic, *content-based* string for a config-field value.

    Crucially, numpy/pandas constants are hashed by their values (not by object
    identity) so an identical feature block gets the same signature across
    pipelines even though each pipeline re-imports its ``features.py`` and mints
    fresh Series objects. (stratum's own ``config_key`` falls back to ``id()`` for
    these unhashable leaves, which is right for in-DAG CSE but wrong across DAGs.)
    """
    if isinstance(v, OperandRef):
        return f"${v.k}"
    np = _np()
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            # object arrays (e.g. string index): tobytes() would hash pointer
            # addresses, so canonicalize by element values instead.
            body = _hb(repr(v.tolist()).encode())
        else:
            body = _hb(np.ascontiguousarray(v).tobytes())
        return f"nd[{v.dtype}{tuple(v.shape)}#{body}]"
    if isinstance(v, np.generic):
        return repr(v.item())
    pd = _pd()
    if isinstance(v, pd.Series):
        return f"ser[{v.dtype}|i={_field_str(v.index.to_numpy())}|v={_field_str(v.to_numpy())}]"
    if isinstance(v, pd.Index):
        return f"ix[{_field_str(v.to_numpy())}]"
    if isinstance(v, pd.DataFrame):
        return f"df[{','.join(map(str, v.columns))}|{_field_str(v.to_numpy())}]"
    if isinstance(v, type):
        return f"{v.__module__}.{v.__qualname__}"
    if isinstance(v, BaseEstimator):
        return _estimator_str(v)
    if isinstance(v, (set, frozenset)):
        return "{" + ",".join(sorted(_field_str(x) for x in v)) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_field_str(x) for x in v) + "]"
    if isinstance(v, dict):
        items = sorted(v.items(), key=lambda kv: str(kv[0]))
        return "{" + ",".join(f"{k}={_field_str(x)}" for k, x in items) + "}"
    return _stable_repr(v)


def _estimator_str(est) -> str:
    dataop = _dataop_type()
    try:
        items = sorted(est.get_params(deep=False).items(), key=lambda kv: kv[0])
    except Exception:
        items = []
    parts = [f"{k}={'<graph>' if isinstance(val, dataop) else _field_str(val)}"
             for k, val in items]
    return f"est[{type(est).__module__}.{type(est).__qualname__}({','.join(parts)})]"


def _config_sig(op) -> str:
    """Signature of an op's own configuration (excluding its inputs)."""
    fields = getattr(type(op), "fields", None)
    if fields is None:
        # Opaque op (ImplOp etc.): stable part of repr, minus the volatile suffix.
        return "repr:" + _stable_repr(op).split("[cloned")[0]
    return ";".join(f"{n}={_field_str(getattr(op, n, None))}" for n in fields)


def _readable_label(op) -> str:
    """Human label matching stratum's own ``show_graph`` convention."""
    try:
        op.update_name()
    except Exception:
        pass
    text = str(op)
    return text if text else type(op).__name__


def _estimator_info(op):
    """(estimator_class, {param: str_value}) for estimator ops, else None.

    DataOp-valued params (graph-fed hyperparameters) are dropped -- their binding
    lives in the op's edges, not in a comparable value.
    """
    if not isinstance(op, BaseEstimatorOp):
        return None
    est = getattr(op, "estimator", None)
    if est is None:
        return None
    dataop = _dataop_type()
    params = {}
    try:
        raw = est.get_params(deep=False)
    except Exception:
        raw = getattr(est, "get_params", lambda: {})()
    for k, v in raw.items():
        if isinstance(v, dataop):
            continue
        params[k] = _short(v)
    return (type(est).__name__, params)


def _short(v, limit=80):
    s = repr(v)
    return s if len(s) <= limit else s[:limit] + "…"


@dataclass
class Node:
    nid: str                 # stable id within this DAG (the content signature)
    op_type: str             # class name, e.g. "NumericOp"
    family: str              # logical family label, e.g. "Predictor"
    label: str               # readable, e.g. "NumericOp(multiply)"
    sig: str                 # recursive content signature (== nid)
    inputs: list             # signatures of input nodes
    is_choice: bool = False
    outcome_names: list = field(default_factory=list)
    estimator: tuple | None = None   # (class_name, params) or None


@dataclass
class Dag:
    root_sig: str
    nodes: dict               # sig -> Node  (deduped by content signature)
    order: list               # sigs in topological order (inputs before op)
    reprs: dict               # sig -> readable label (convenience)

    def histogram(self) -> dict:
        h = {}
        for n in self.nodes.values():
            h[n.op_type] = h.get(n.op_type, 0) + 1
        return h

    def estimators(self) -> list:
        return [n for n in self.nodes.values() if n.estimator is not None]


def build_dag(root) -> Dag:
    """Walk a stratum logical Op DAG into a signature-keyed :class:`Dag`."""
    sig_memo: dict[int, str] = {}

    def sig(op) -> str:
        if id(op) in sig_memo:
            return sig_memo[id(op)]
        child_sigs = [sig(i) for i in op.inputs]
        payload = "|".join([
            type(op).__name__,
            _config_sig(op),
            "(" + ",".join(child_sigs) + ")",
        ])
        s = hashlib.sha1(payload.encode()).hexdigest()[:16]
        sig_memo[id(op)] = s
        return s

    nodes: dict[str, Node] = {}
    order: list[str] = []
    reprs: dict[str, str] = {}
    for op in topological_iterator(root):
        s = sig(op)
        if s not in nodes:
            outcome = []
            if isinstance(op, ChoiceOp):
                try:
                    outcome = list(op.make_outcome_names())
                except Exception:
                    outcome = list(getattr(op, "outcome_names", []) or [])
            label = _readable_label(op)
            nodes[s] = Node(
                nid=s,
                op_type=type(op).__name__,
                family=getattr(type(op), "logical_family", None) or type(op).__name__,
                label=label,
                sig=s,
                inputs=[sig(i) for i in op.inputs],
                is_choice=isinstance(op, ChoiceOp),
                outcome_names=outcome,
                estimator=_estimator_info(op),
            )
            reprs[s] = label
            order.append(s)

    return Dag(root_sig=sig(root), nodes=nodes, order=order, reprs=reprs)
