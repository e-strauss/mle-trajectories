"""Validate a candidate skrubified pipeline: static checks + a real plan build.

The plan build runs in a SUBPROCESS (``_validate_child.py``) so a candidate that
imports heavy libraries, mutates globals, parses argv or calls ``sys.exit`` can
never disturb the tool, and so the plan can be built by a different interpreter
(``--python``) than the one running skrubify. It needs no dataset: the child
forces ``eager_data_ops=False``, so the recorded read is never executed.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .checks import CheckReport, run_checks

CHILD = Path(__file__).with_name("_validate_child.py")


MISSING_MODULE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")


@dataclass
class Validation:
    checks: CheckReport
    build_ok: bool | None = None          # None => build skipped or not attempted
    build_error: str | None = None
    info: dict = field(default_factory=dict)   # n_nodes / has_X / has_y / cv / …
    missing_module: str | None = None     # dependency gap, NOT a bug in the candidate

    @property
    def ok(self) -> bool:
        return self.checks.ok and self.build_ok is not False and not self.structural

    @property
    def structural(self) -> list[str]:
        """Problems visible in the built graph (not in the text)."""
        out = []
        if self.build_ok:
            if self.info.get("has_X") is False:
                out.append("the built plan has no node marked as X "
                           "(mark_as_X is missing or on a discarded branch).")
            if self.info.get("has_y") is False:
                out.append("the built plan has no node marked as y.")
            if self.info.get("has_X") and not self.info.get("cv"):
                out.append("the X node carries no CV splitter -- pass cv=... to "
                           "mark_as_X.")
        return out

    def feedback(self) -> str:
        """The text handed back to the model for a repair round."""
        parts = []
        if self.checks.errors:
            parts.append("Static checks failed:\n" +
                         "\n".join(f"- {m}" for m in self.checks.errors))
        if self.build_error and not self.missing_module:
            parts.append("Building the plan failed (imported with "
                         "eager_data_ops=False, no data read):\n```\n"
                         f"{self.build_error}\n```")
        if self.structural:
            parts.append("The plan builds but its graph is wrong:\n" +
                         "\n".join(f"- {m}" for m in self.structural))
        if self.checks.warnings:
            parts.append("Warnings (fix if they are real, ignore if intentional):\n" +
                         "\n".join(f"- {m}" for m in self.checks.warnings))
        return "\n\n".join(parts)

    def summary(self) -> str:
        bits = []
        if self.missing_module:
            bits.append(f"build SKIPPED: {self.missing_module!r} is not installed in "
                        "the validating interpreter "
                        f"(pip install {self.missing_module}, or use --python)")
        elif self.build_ok is None:
            bits.append("build skipped")
        elif self.build_ok:
            info = self.info
            bits.append(f"plan builds ({info.get('n_nodes', '?')} nodes)")
            if info.get("cv"):
                bits.append(f"cv={info['cv']}")
            grid = info.get("param_grid")
            if grid and "empty" not in grid.lower():
                bits.append(f"grid: {grid.splitlines()[0]}…")
        else:
            bits.append("plan build FAILED")
        bits.append(f"{len(self.checks.errors)} check error(s), "
                    f"{len(self.checks.warnings)} warning(s)")
        return " · ".join(bits)


def build_plan(path: Path, python: str | None = None, timeout: int = 120) -> tuple[bool, str | None, dict]:
    """Import ``path`` in a subprocess and return (ok, error, info)."""
    cmd = [python or sys.executable, str(CHILD), str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"building the plan timed out after {timeout}s", {}
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        report = json.loads(line)
    except json.JSONDecodeError:
        detail = (proc.stderr or proc.stdout or "").strip()[-4000:]
        return False, f"the validator subprocess crashed:\n{detail}", {}
    ok = bool(report.pop("ok", False))
    error = report.pop("error", None)
    if not ok and not error:
        error = f"plan build failed at stage {report.get('stage')!r}"
    return ok, error, report


def validate(path: Path, *, python: str | None = None, strict: bool = False,
             build: bool = True, timeout: int = 120) -> Validation:
    source = Path(path).read_text()
    checks = run_checks(source, strict=strict)
    if not build:
        return Validation(checks=checks)
    ok, error, info = build_plan(Path(path), python=python, timeout=timeout)
    # A third-party module missing from the validating interpreter is an
    # environment gap, not a defect in the candidate: reporting it as a build
    # failure pushes the model into dropping the estimator or hiding the import
    # inside fit(). Surface it to the user instead and leave the build unjudged.
    missing = None
    if not ok and error:
        m = MISSING_MODULE.search(error)
        if m and m.group(1).split(".")[0] not in ("skrub", "pandas", "numpy", "sklearn"):
            missing, ok = m.group(1), None
    return Validation(checks=checks, build_ok=ok, build_error=error, info=info,
                      missing_module=missing)


# Every pattern is applied and the matches pooled: a script may print its
# variants one way ("Ablation 1 ... Accuracy: 0.9482") and a final summary
# another ("Final Validation Performance: 0.94886"), and picking only the first
# matching pattern would keep the summary and miss all the variants. Repeated or
# rounded restatements of the same number are harmless -- the comparison only
# asks that every score the CONVERSION produced is found among them.
SCORE_PATTERNS = (
    r"Final Validation (?:Performance|Score)\s*:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)",
    r"mean_test_score\s*:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)",
    r"(?:CV|cross-validat\w+)[^\n:]*:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)",
    r"(?i)(?:performance|accuracy|score|r2|rmse|auc)[^\n:]*:\s*"
    r"(-?[\d.]+(?:[eE][-+]?\d+)?)",
)


VARIANT_PATTERN = r"Variant score:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)"


def parse_score(stdout: str) -> float | None:
    """The score a pipeline printed, or None. Last match wins (final line)."""
    for pattern in SCORE_PATTERNS:
        found = re.findall(pattern, stdout)
        if found:
            return float(found[-1])
    return None


def parse_scores(stdout: str) -> list[float]:
    """EVERY score in the output, in print order.

    A converted plan with choices prints one `Variant score:` line per grid row,
    which is what a multi-variant original (an ablation study printing a score per
    experiment) has to be compared against -- comparing one number from each side
    picks the original's LAST experiment against the plan's BEST variant, which is
    meaningless.
    """
    return parse_scores_with_precision(stdout)[0]


def parse_scores_with_precision(stdout: str) -> tuple[list[float], float]:
    """Scores plus the tolerance implied by how many decimals they were PRINTED to.

    Scripts commonly print `f"{score:.4f}"`, so 0.94727 reaches us as "0.9473".
    Comparing that against a full-precision score at 1e-6 reports a spurious
    mismatch; the honest tolerance is half of the last printed digit.
    """
    raw = re.findall(VARIANT_PATTERN, stdout)
    if not raw:
        # Pool every pattern, but keep each NUMBER once: two patterns matching the
        # same text (a "Final Validation Performance:" line also matches the
        # generic one) must not count as two scores. Dedupe by position in the
        # output, so the same value printed twice in two places still counts twice.
        seen: dict[int, str] = {}
        for pattern in SCORE_PATTERNS:
            for m in re.finditer(pattern, stdout):
                seen.setdefault(m.start(1), m.group(1))
        raw = [seen[k] for k in sorted(seen)]
    if not raw:
        return [], 1e-6
    decimals = min((len(t.split(".")[1]) if "." in t else 0) for t in raw)
    return [float(t) for t in raw], max(1e-6, 0.5 * 10 ** -decimals)


def compare_scores(new: list[float], old: list[float], tol: float = 1e-6) -> str:
    """Human-readable verdict over two score lists, order-independent."""
    if not new or not old:
        return "no comparable scores"
    if len(new) == 1 and len(old) == 1:
        delta = new[0] - old[0]
        # skrub scores with sklearn's higher-is-better convention, so an error
        # metric comes back NEGATED (neg_root_mean_squared_error). The original
        # script prints the plain positive error. Compare magnitudes in that case
        # instead of reporting a spurious 2x delta.
        flipped = new[0] * old[0] < 0 and abs(abs(new[0]) - abs(old[0])) < abs(delta)
        if flipped:
            delta = abs(new[0]) - abs(old[0])
        verdict = ("identical" if delta == 0 else
                   "close" if abs(delta) < 1e-3 else "DIFFERENT")
        note = "  [sign-flipped: neg_* scorer vs positive error metric]" if flipped else ""
        return f"{new[0]!r} vs {old[0]!r}  delta {delta:+.6g}  ({verdict}){note}"
    matched, unmatched = 0, []
    remaining = list(old)
    for value in new:
        hit = next((o for o in remaining if abs(o - value) <= tol), None)
        if hit is None:
            unmatched.append(value)
        else:
            remaining.remove(hit)
            matched += 1
    bits = [f"{matched}/{len(new)} variant scores found in the original's "
            f"{len(old)} printed number(s) within {tol:g}"]
    if unmatched:
        bits.append(f"NOT FOUND in the original: {[round(v, 6) for v in unmatched]}")
    if remaining:
        # Usually just the same scores restated (a summary block, or a rounded
        # and a full-precision copy of one number) -- not evidence of a problem.
        bits.append(f"({len(remaining)} other number(s) printed by the original)")
    return "; ".join(bits)


def run_pipeline(path: Path, cwd: Path, *, python: str | None = None,
                 timeout: int = 1800) -> tuple[bool, float | None, str]:
    """Actually execute a pipeline and read the score it prints.

    This is the ONLY layer that catches scoring-time failures -- a plan can build
    perfectly and still die inside make_grid_search (e.g. skrub < 0.10 with
    split_kwargs=None), and only a real run compares a conversion's score against
    the original's. Needs the dataset: `cwd` must be a directory the script's own
    relative paths resolve against (typically one containing ./input).
    """
    cmd = [python or sys.executable, str(Path(path).resolve())]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=str(cwd))
    except subprocess.TimeoutExpired:
        return False, None, f"the run timed out after {timeout}s"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-4000:]
        return False, None, f"exit code {proc.returncode}:\n{tail}"
    score = parse_score(proc.stdout)
    if score is None:
        return False, None, ("the run finished but printed no recognisable score "
                             f"(expected 'Final Validation Performance: <number>'):\n"
                             f"{proc.stdout.strip()[-1500:]}")
    return True, score, proc.stdout.strip()[-1500:]
