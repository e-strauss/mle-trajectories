"""Run inside the target interpreter: build a candidate pipeline's plan, report JSON.

    python -m skrubify._validate_child <pipeline.py>       (or: python _validate_child.py …)

Prints ONE json object on stdout. Never raises: a failure is reported in the
"error" field. Building the plan touches no data -- we force
``eager_data_ops=False`` so recorded reads/previews are not executed, which is
what lets the tool validate a pipeline with no dataset present.
"""
from __future__ import annotations

import contextlib
import io
import json
import runpy
import sys
import traceback
from pathlib import Path

MAX_STDOUT = 2000
TB_LIMIT = 12


def _fmt_exc(exc: BaseException) -> str:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=TB_LIMIT))
    return tb.strip()


def _marks(pred):
    """(has_X, has_y, cv_repr) -- best effort, tolerant of skrub internals moving."""
    has_x = has_y = False
    cv = None
    try:
        from skrub._data_ops import _evaluation

        for node in _evaluation.nodes(pred):
            impl = getattr(node, "_skrub_impl", None)
            if impl is None:
                continue
            if getattr(impl, "is_X", False):
                has_x = True
                if getattr(impl, "cv", None) is not None:
                    cv = repr(impl.cv)
            if getattr(impl, "is_y", False):
                has_y = True
    except Exception:  # pragma: no cover - introspection is advisory only
        return None, None, None
    return has_x, has_y, cv


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = Path(argv[0]).resolve()
    out: dict = {"ok": False, "stage": "start", "path": str(path)}

    try:
        import skrub
    except Exception as exc:
        out.update(stage="import-skrub", error=_fmt_exc(exc))
        print(json.dumps(out))
        return 1
    out["skrub_version"] = getattr(skrub, "__version__", "?")

    # The candidate is a standalone script: give it a bare argv (its own argparse
    # must not see ours) and run it under a non-"__main__" name so its
    # `if __name__ == "__main__"` scoring block stays dormant.
    sys.argv = [str(path)]
    sys.path.insert(0, str(path.parent))
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            with skrub.config_context(eager_data_ops=False):
                ns = runpy.run_path(str(path), run_name="__skrubify_validate__")
                out["stage"] = "find-pred"
                pred = ns.get("pred")
                if pred is None:
                    raise AttributeError(
                        "the module defines no module-level `pred` "
                        "(the final prediction DataOp)"
                    )
                if not hasattr(pred, "skb"):
                    raise TypeError(
                        f"module-level `pred` is a {type(pred).__name__}, not a skrub "
                        "DataOp -- it must be the node returned by .skb.apply(model, y=y)"
                    )
                out["stage"] = "inspect"
                has_x, has_y, cv = _marks(pred)
                out.update(has_X=has_x, has_y=has_y, cv=cv)
                with contextlib.suppress(Exception):
                    out["n_nodes"] = len(__import__(
                        "skrub._data_ops._evaluation", fromlist=["nodes"]).nodes(pred))
                with contextlib.suppress(Exception):
                    out["param_grid"] = str(pred.skb.describe_param_grid()).strip()
                with contextlib.suppress(Exception):
                    out["steps"] = str(pred.skb.describe_steps()).strip()
        out.update(ok=True, stage="done")
    except SystemExit as exc:
        out.update(stage=out["stage"], error=f"the script called sys.exit({exc.code}) "
                                             "while building the plan")
    except BaseException as exc:  # noqa: BLE001 - report anything the plan raises
        out["error"] = _fmt_exc(exc)
    finally:
        text = buf.getvalue()
        if text:
            out["stdout"] = text[-MAX_STDOUT:]
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
