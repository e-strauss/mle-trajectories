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


SCORE_PATTERNS = (
    r"Final Validation (?:Performance|Score)\s*:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)",
    r"mean_test_score\s*:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)",
    r"(?:CV|cross-validat\w+)[^\n:]*:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)",
)


def parse_score(stdout: str) -> float | None:
    """The score a pipeline printed, or None. Last match wins (final line)."""
    for pattern in SCORE_PATTERNS:
        found = re.findall(pattern, stdout)
        if found:
            return float(found[-1])
    return None


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
