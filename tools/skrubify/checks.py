"""Cheap static checks on a candidate skrubified pipeline.

They encode the output contract and the guide's pitfalls (skrub_dataops_summary.md
section 12) as regexes over the source text. They run before the (slower) plan
build, and every failure is fed back to the model verbatim as repair feedback.

Each :class:`Rule` is either a "require" (the pattern MUST match somewhere) or a
"forbid" (it must NOT match). Severity ``error`` triggers a repair round;
``warning`` is reported but accepted.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str          # "require" | "forbid"
    pattern: str
    message: str
    level: str = ERROR
    # Region where a violation is an ERROR. Outside it, a "forbid" rule still
    # reports, but only as a warning:
    #   "all"      -- the whole file
    #   "plan"     -- inside the `with skrub.config_context(...)` block
    #   "toplevel" -- module level (incl. the plan block), i.e. outside every
    #                 def/class body, where ordinary pandas/sklearn code is fine
    scope: str = "all"

    def check(self, source: str) -> list[str]:
        rx = re.compile(self.pattern, re.MULTILINE)
        if self.kind == "require":
            return [] if rx.search(source) else [self.message]
        return [f"line {source[:m.start()].count(chr(10)) + 1}: {m.group(0).strip()[:80]}"
                f" -- {self.message}" for m in rx.finditer(source)]


RULES: tuple[Rule, ...] = (
    # --- the output contract -------------------------------------------------
    Rule("import-skrub", "require", r"\bimport\b[^\n]*\bskrub\b",
         "the file must `import skrub`."),
    Rule("recorded-read", "require", r"skrub\.(as_data_op|var|X|y)\s*\(",
         "the data load must be recorded in the plan: "
         "`skrub.as_data_op(path).skb.apply_func(pd.read_csv)` "
         "(or skrub.var(...)), never a bare module-level pd.read_csv."),
    Rule("mark-x", "require", r"\.skb\.mark_as_X\s*\(",
         "the design matrix must be marked with `.skb.mark_as_X(cv=...)`."),
    Rule("mark-y", "require", r"\.skb\.mark_as_y\s*\(",
         "the target must be marked with `.skb.mark_as_y()`."),
    Rule("module-pred", "require", r"^\s*pred\s*=",
         "the final prediction node must be assigned to a module-level name `pred`."),
    Rule("eager-off", "require",
         r"config_context\s*\(\s*eager_data_ops\s*=\s*False",
         "wrap the whole plan construction in "
         "`with skrub.config_context(eager_data_ops=False):` so importing the file "
         "builds the graph without touching data."),
    Rule("main-guard", "require",
         r"""if\s+__name__\s*==\s*['"]__main__['"]\s*:""",
         'the scoring block must be guarded by `if __name__ == "__main__":` so '
         "importing the file builds the plan without reading data or fitting a "
         "model."),
    Rule("no-argparse", "forbid", r"\bargparse\b|\badd_argument\s*\(",
         "the script takes no command-line arguments -- drop argparse entirely "
         "and guard the scoring block with `if __name__ == \"__main__\":`."),

    # --- leftovers from the original script ----------------------------------
    Rule("manual-split", "forbid", r"\btrain_test_split\s*\(",
         "the manual train/validation split is replaced by the CV splitter on "
         "mark_as_X (use ShuffleSplit(n_splits=1, test_size=...) to mirror a single "
         "hold-out split).", ERROR, "toplevel"),
    Rule("manual-cv-loop", "forbid",
         r"^\s*for\s+.*\bin\b.*\b\w*(skf|kf|cv|splitter|split)\w*\.split\s*\(",
         "the manual fold loop must be removed -- skrub re-runs the whole recorded "
         "plan per fold.", ERROR, "toplevel"),
    Rule("fold-indexing", "forbid", r"\.iloc\s*\[\s*(train|val|test|valid)_(index|idx)",
         "per-fold row indexing must be removed; the CV splitter owns the splits.", ERROR, "toplevel"),
    Rule("sklearn-cv-helper", "forbid", r"\b(cross_val_score|cross_val_predict)\s*\(",
         "score through make_grid_search on the plan, not sklearn's cross_val_* "
         "helpers.", ERROR, "toplevel"),
    Rule("manual-metric", "forbid",
         r"\b(accuracy_score|mean_squared_error|mean_absolute_error|r2_score|"
         r"roc_auc_score|f1_score|log_loss|balanced_accuracy_score)\s*\(",
         "the metric is expressed as make_grid_search(scoring=\"...\"), not computed "
         "by hand.", WARNING, "toplevel"),
    Rule("submission", "forbid", r"\.to_(csv|parquet)\s*\(",
         "writing predictions/intermediate tables is out of scope -- the deliverable "
         "is a cross-validated score."),
    Rule("test-csv", "forbid", r"""["']?[^"'\s]*test\.csv""",
         "the test set is not part of a scored plan; drop it.", WARNING),
    Rule("chunked-read", "forbid", r"\bchunksize\s*=",
         "a chunked read cannot be recorded as one node -- read the table once "
         "(add .skb.subsample(n=...) for cheap previews)."),
    Rule("makedirs", "forbid", r"\bos\.makedirs\s*\(",
         "no output directories are needed.", WARNING),
    Rule("early-stopping", "forbid", r"\b(early_stopping_rounds|eval_set)\s*=",
         "at plan level there is no eval set to hand the booster -- drop early "
         "stopping, or keep it inside a wrapper estimator that makes its own "
         "inner validation split in fit().", ERROR, "toplevel"),

    # --- guide pitfalls ------------------------------------------------------
    Rule("apply-plain-func", "forbid",
         r"\.skb\.apply\s*\(\s*(np\.|numpy\.|pd\.|pandas\.|lambda\b)",
         ".skb.apply is for scikit-learn estimators only; a plain function goes "
         "through .skb.apply_func / skrub.deferred."),
    Rule("sparse-ohe", "forbid",
         r"OneHotEncoder\s*\((?![^)]*sparse_output\s*=\s*False)[^)]*\)",
         "skrub works on dense data: pass sparse_output=False to OneHotEncoder.",
         WARNING),
    Rule("column-loop", "forbid", r"^\s*for\s+\w+\s+in\s+\w*[Xx]\w*\.columns\b",
         "you cannot loop over columns inside a plan -- use skrub.selectors plus "
         "transformer broadcasting (guide section 6).", ERROR, "plan"),
    Rule("continuous-choice", "forbid", r"\bchoose_(float|int)\s*\(",
         "grid search enumerates DISCRETE values only -- use "
         "skrub.choose_from([...]) with explicit values."),
    Rule("randomized-search", "forbid", r"\bmake_randomized_search\s*\(",
         "score with make_grid_search, not make_randomized_search."),
    Rule("draw-graph-open", "forbid", r"draw_graph\s*\(\s*\)\s*\.open\s*\(",
         "do not open a graph viewer -- it blocks and needs a browser."),
    Rule("deferred-block", "forbid", r"^\s*@skrub\.deferred\b",
         "prefer fine-grained recorded ops over one opaque @skrub.deferred node; "
         "reserve deferred for a single non-recordable call (guide section 4).",
         WARNING),
    Rule("copy-df", "forbid", r"\.copy\s*\(\s*\)",
         "recorded ops are immutable; .copy() is a leftover from in-place pandas "
         "mutation.", WARNING, "plan"),
    Rule("inplace-assign", "forbid",
         r"^\s*(df|X|X_train|data|train_df)\s*\[\s*[\"'][^\"']+[\"']\s*\]\s*=(?!=)",
         "in-place column assignment is not recorded -- use .assign(...) instead.",
         ERROR, "plan"),
)


@dataclass
class CheckReport:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_text(self) -> str:
        lines = [f"- ERROR: {m}" for m in self.errors]
        lines += [f"- warning: {m}" for m in self.warnings]
        return "\n".join(lines)


def plan_block(source: str) -> str:
    """Mask everything outside the ``with skrub.config_context(...)`` block.

    Line numbers are preserved. Some rules must only look at the plan itself: a
    custom estimator or a helper function in the same file is ORDINARY pandas
    code, where `.copy()`, in-place assignment and `for col in X.columns` are all
    legitimate -- they are only wrong among recorded ops. If no plan block is
    found, everything is in scope.
    """
    lines = source.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if re.search(r"^\s*with\s+.*config_context\s*\(", line):
            start = i
            indent = len(line) - len(line.lstrip())
            break
    if start is None:
        return source
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if stripped and (len(lines[j]) - len(lines[j].lstrip())) <= indent:
            end = j
            break
    return "".join(l if start <= k < end else "\n" * l.count("\n")
                   for k, l in enumerate(lines))


def mask_functions(source: str) -> str:
    """Mask the body of every function/method/class, preserving line numbers.

    A manual fold loop, a `train_test_split` or a hand-computed metric is a real
    defect at module level -- but perfectly legitimate INSIDE a custom estimator
    (an inner validation split for early stopping, a scorer's own arithmetic).
    Outside this region such matches degrade to warnings instead of forcing a
    repair round that would delete working code.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines(keepends=True)
    keep = [True] * (len(lines) + 1)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for row in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                if row <= len(lines):
                    keep[row] = False
    return "".join(l if keep[i + 1] else "\n" * l.count("\n")
                   for i, l in enumerate(lines))


def _complement(source: str, masked: str) -> str:
    """The lines `masked` blanked out -- where a scoped rule is only advisory."""
    src_lines = source.splitlines(keepends=True)
    msk_lines = masked.splitlines(keepends=True)
    return "".join(l if (i >= len(msk_lines) or not msk_lines[i].strip()) else
                   "\n" * l.count("\n") for i, l in enumerate(src_lines))


def _calls(source: str, name: str):
    """Yield ``(lineno, args_text)`` per ``name(...)`` call, brace-balanced.

    Needed because an argument list can contain nested calls
    (``make_grid_search(cv=KFold(3), fitted=True)``), which a flat ``[^)]*``
    regex cannot see past.
    """
    for m in re.finditer(rf"\b{re.escape(name)}\s*\(", source):
        i, depth = m.end(), 1
        while i < len(source) and depth:
            c = source[i]
            depth += (c in "([{") - (c in ")]}")
            i += 1
        yield source[: m.start()].count("\n") + 1, source[m.end(): i - 1]


def _target_transform_check(source: str) -> list[tuple[str, str]]:
    """Fit on a transformed target, but never invert it on the predictions?

    The CV score is computed against the node marked by ``mark_as_y``. Fitting on
    a target derived from it (``y = y_raw - 1``, ``np.log1p``, ``astype``, ...)
    without mapping the predictions back therefore scores predictions in the
    WRONG DOMAIN -- silently, with a plan that builds fine and a plausible-looking
    score (e.g. accuracy 0.0 for a label shift). The inverse must be applied to
    the prediction node gated on ``skrub.eval_mode()`` (guide section 8).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    def names_in(node) -> set[str]:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    def marks_y(node) -> bool:
        return any(isinstance(n, ast.Attribute) and n.attr == "mark_as_y"
                   for n in ast.walk(node))

    marked: set[str] = set()       # names holding the marked (raw) target
    derived: set[str] = set()      # names holding a TRANSFORM of the marked target
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or node.value is None:
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if marks_y(node.value):
            marked |= targets
        elif names_in(node.value) & (marked | derived) and not isinstance(
                node.value, (ast.Name, ast.Attribute)):
            derived |= targets

    gated = ("eval_mode" in source) or (".skb.match(" in source)
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "apply"):
            continue
        for kw in node.keywords:
            if kw.arg != "y":
                continue
            transformed = (isinstance(kw.value, ast.Name) and kw.value.id in derived) or (
                not isinstance(kw.value, ast.Name)
                and names_in(kw.value) & (marked | derived))
            if transformed and not gated:
                out.append((ERROR, f"line {node.lineno}: the model is fitted on a "
                            "TRANSFORMED target while the score is computed against "
                            "the mark_as_y node, so the predictions end up in the "
                            "wrong domain. Either mark the transformed target "
                            "directly, or map the predictions back with "
                            "`pred.skb.apply_func(inverse, skrub.eval_mode())` "
                            "returning the input unchanged in \"fit\" mode "
                            "(guide section 8)."))
    return out


def _udf_check(source: str) -> list[tuple[str, str]]:
    """Advisory: `apply_func(<own function>)` that is probably expressible as ops.

    `apply_func` / `deferred` collapse whatever they wrap into ONE opaque node
    (guide pitfall 13). Two uses are legitimate and skipped here: a library call
    that cannot be recorded (`pd.read_csv`, `np.sqrt`, `pd.to_datetime`) and an
    eval-mode-gated post-prediction step (`apply_func(f, skrub.eval_mode())`).
    A locally-defined dataframe->dataframe helper is usually neither: pandas
    filtering, arithmetic, assignment, `value_counts`/`isin`/boolean masks all
    record fine (guide section 4). Warning only -- some helpers genuinely cannot
    be recorded.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    local_funcs = {n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("apply_func", "deferred")):
            continue
        if "eval_mode" in ast.dump(node):        # gated post-prediction: fine
            continue
        first = node.args[0] if node.args else None
        name = (first.id if isinstance(first, ast.Name) else
                "<lambda>" if isinstance(first, ast.Lambda) else None)
        if name and (name in local_funcs or name == "<lambda>"):
            out.append((WARNING, f"line {node.lineno}: {node.func.attr}({name}) wraps "
                        "your own function into ONE opaque plan node -- if its body "
                        "is pandas (filtering, assignment, value_counts/isin, "
                        "arithmetic), write it as recorded ops instead so each step "
                        "is its own node (guide section 4)."))
    return out


def _arg_checks(source: str) -> list[tuple[str, str]]:
    """(level, message) pairs from checks over parsed call arguments."""
    out: list[tuple[str, str]] = []

    for lineno, args in _calls(source, "mark_as_X"):
        if not re.search(r"\bcv\s*=", args):
            out.append((ERROR, f"line {lineno}: mark_as_X() carries no cv= -- the CV "
                               "splitter must be passed here, the only place it can live."))
        elif not re.search(r"\bsplit_kwargs\s*=", args):
            # skrub < 0.10 keeps a missing split_kwargs as None and later does
            # `**None`: the plan builds, then dies inside make_grid_search.
            out.append((ERROR, f"line {lineno}: pass split_kwargs={{}} next to cv= on "
                               "mark_as_X (or a real dict for a grouped splitter). "
                               "Without it skrub < 0.10 raises `TypeError: argument "
                               "after ** must be a mapping, not NoneType` at scoring "
                               "time, long after the plan builds cleanly."))

    seen_search = False
    for lineno, args in _calls(source, "make_grid_search"):
        seen_search = True
        if re.search(r"\bcv\s*=", args):
            out.append((ERROR, f"line {lineno}: a cv= passed to make_grid_search "
                               "OVERRIDES mark_as_X(cv=...) -- pass none here "
                               "(guide section 3)."))
        if not re.search(r"\bfitted\s*=\s*True", args):
            out.append((ERROR, f"line {lineno}: make_grid_search must be called with "
                               "fitted=True (that is what runs the cross-validation)."))
        if not re.search(r"\brefit\s*=\s*False", args):
            out.append((ERROR, f"line {lineno}: make_grid_search must be called with "
                               "refit=False (the full-data refit is not needed)."))
        if not re.search(r"\bscoring\s*=", args):
            out.append((ERROR, f"line {lineno}: pass scoring=\"<sklearn scorer>\" to "
                               "make_grid_search, matching the original's metric."))
    if not seen_search:
        out.append((ERROR, "score the plan with `pred.skb.make_grid_search(n_jobs=1, "
                           "fitted=True, refit=False, scoring=...)`."))

    out += _target_transform_check(source)
    out += _udf_check(source)

    for lineno, args in _calls(source, ".skb.concat"):
        if args.strip() and not args.lstrip().startswith("["):
            out.append((ERROR, f"line {lineno}: .skb.concat takes a LIST of others: "
                               "a.skb.concat([b], axis=1)."))
    return out


def strip_noncode(source: str) -> str:
    """Blank out comments and multi-line (doc)strings, preserving line numbers.

    Rules must fire on real code only: a comment saying "the submission code was
    dropped" or a docstring quoting `train_test_split` is not a violation. Single
    -line string literals are kept, since some rules match path literals.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    lines = source.splitlines(keepends=True)
    for tok in tokens:
        if tok.type == tokenize.COMMENT or (
                tok.type == tokenize.STRING and tok.end[0] > tok.start[0]):
            (srow, scol), (erow, ecol) = tok.start, tok.end
            for row in range(srow, erow + 1):
                idx = row - 1
                if idx >= len(lines):
                    break
                line = lines[idx]
                start = scol if row == srow else 0
                end = ecol if row == erow else len(line.rstrip("\n"))
                lines[idx] = line[:start] + " " * (end - start) + line[end:]
    return "".join(lines)


def run_checks(source: str, *, strict: bool = False) -> CheckReport:
    source = strip_noncode(source)
    regions = {"all": source, "plan": plan_block(source),
               "toplevel": mask_functions(source)}
    errors: list[str] = []
    warnings: list[str] = []
    for rule in RULES:
        region = regions[rule.scope]
        for msg in rule.check(region):
            if rule.level == ERROR or strict:
                errors.append(msg)
            else:
                warnings.append(msg)
        if rule.kind == "forbid" and rule.scope != "all" and rule.level == ERROR:
            # an error-level rule outside its scope: report, never force a repair
            for msg in rule.check(_complement(source, region)):
                warnings.append(msg)
    for level, msg in _arg_checks(source):
        (errors if level == ERROR or strict else warnings).append(msg)
    return CheckReport(errors, warnings)
