#!/usr/bin/env python3
"""Tabular overview of an MLE development-process trajectory.

Usage:
    python tools/trajectory.py path/to/final_state.json
    python tools/trajectory.py path/to/final_state.json --type mle-star
    python tools/trajectory.py path/to/final_state.json --no-color --sort time
    python tools/trajectory.py path/to/final_state.json --modules

The trajectory *parsers* live in ``pipeline_analyzer/trajectory.py`` (shared with
the pipeline analyzer, which builds pipeline lineage from the same steps); this
file is the renderer and CLI. To support a new run format, add a parser there.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_parsers():
    """Import the parser module by path, so this stays a dependency-free script
    (importing the ``pipeline_analyzer`` package would pull in skrub/stratum)."""
    path = Path(__file__).resolve().parent / "pipeline_analyzer" / "trajectory.py"
    spec = importlib.util.spec_from_file_location("_mle_trajectory", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses resolves annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


_traj = _load_parsers()
PARSERS = _traj.PARSERS
Step, Trajectory = _traj.Step, _traj.Trajectory


# --------------------------------------------------------------------------- #
# Rendering (format-agnostic)
# --------------------------------------------------------------------------- #
class C:
    """Terminal colours; blanked out when colour is disabled."""
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"

    @classmethod
    def disable(cls):
        for name in ("RESET", "BOLD", "DIM", "CYAN", "GREEN", "YELLOW", "RED"):
            setattr(cls, name, "")


def _fmt_time(sec: float | None) -> str:
    if sec is None:
        return "-"
    if sec >= 3600:
        return f"{sec / 3600:.2f}h"
    if sec >= 60:
        return f"{sec / 60:.1f}m"
    return f"{sec:.1f}s"


def _fmt_score(score: float | None) -> str:
    return "-" if score is None else f"{score:.5f}"


def render(traj: Trajectory, path: str, sort: str = "order") -> str:
    out: list[str] = []
    width = 72
    bar = "=" * width
    rule = "-" * width

    title = str(traj.meta.get("task", "trajectory"))
    ttype = traj.meta.get("task type", "") or traj.meta.get("format", "")
    out.append(f"{C.BOLD}{C.CYAN}{bar}{C.RESET}")
    header = f" MLE trajectory  —  {title}"
    if ttype:
        header += f"  ({ttype})"
    out.append(f"{C.BOLD}{C.CYAN}{header}{C.RESET}")
    out.append(f"{C.BOLD}{C.CYAN}{bar}{C.RESET}")

    # file path on its own line (usually too long to share a column)
    out.append(f" {C.DIM}{'file':<18}{C.RESET} {path}")

    # remaining meta as two columns
    items = [(k, v) for k, v in traj.meta.items() if k not in ("task", "task type")]
    for i in range(0, len(items), 2):
        left = items[i]
        cell_l = f" {C.DIM}{left[0]:<18}{C.RESET} {left[1]}"
        if i + 1 < len(items):
            right = items[i + 1]
            cell_l = f" {C.DIM}{left[0]:<18}{C.RESET} {str(left[1]):<24}"
            cell_l += f"{C.DIM}{right[0]:<18}{C.RESET} {right[1]}"
        out.append(cell_l)

    # summary
    scored = [s for s in traj.steps if s.score is not None]
    total_time = sum(s.time_s for s in traj.steps if s.time_s)
    failures = sum(1 for s in traj.steps if s.returncode not in (0, None))
    best = traj.best_step()
    out.append(rule)
    direction = "lower-is-better" if traj.lower_is_better else "higher-is-better"
    out.append(f" {C.DIM}{'executions':<18}{C.RESET} {len(traj.steps):<24}"
               f"{C.DIM}{'failures':<18}{C.RESET} {failures}")
    out.append(f" {C.DIM}{'scored runs':<18}{C.RESET} {len(scored):<24}"
               f"{C.DIM}{'metric':<18}{C.RESET} {direction}")
    out.append(f" {C.DIM}{'total exec time':<18}{C.RESET} {_fmt_time(total_time)} "
               f"({total_time:.0f}s)")
    if best is not None:
        out.append(f" {C.DIM}{'best score':<18}{C.RESET} "
                   f"{C.GREEN}{_fmt_score(best.score)}{C.RESET} "
                   f"({best.phase} · {best.ident})")
    submit = next((s for s in traj.steps if s.phase == "Submit"), None)
    if submit is not None:
        out.append(f" {C.DIM}{'final submission':<18}{C.RESET} "
                   f"{_fmt_score(submit.score)}  (rc={submit.returncode})")
    out.append(rule)

    # trajectory table
    steps = list(traj.steps)
    if sort == "time":
        steps.sort(key=lambda s: (s.time_s or 0), reverse=True)
    elif sort == "score":
        steps.sort(key=lambda s: (-(s.score if s.score is not None else -1e9)))

    best_id = (best.phase, best.ident) if best else None
    out.append(f" {C.BOLD}{'#':>3}  {'PHASE':<10} {'ID':<16} "
               f"{'SCORE':>9}  {'TIME':>7}  {'RC':>2}  NOTE{C.RESET}")
    for i, s in enumerate(steps, 1):
        is_best = best_id == (s.phase, s.ident)
        rc_col = C.GREEN if s.returncode == 0 else C.RED
        score_str = _fmt_score(s.score)
        if is_best:
            score_str = f"{C.GREEN}{C.BOLD}{score_str}{C.RESET}"
        elif s.score is None:
            score_str = f"{C.DIM}{score_str}{C.RESET}"
        note = s.note
        if note == "FAILED":
            note = f"{C.RED}FAILED{C.RESET}"
        elif note:
            note = f"{C.DIM}{note}{C.RESET}"
        marker = f"{C.GREEN}★{C.RESET}" if is_best else " "
        # pad score manually because of colour codes
        raw_score = _fmt_score(s.score)
        pad = " " * max(0, 9 - len(raw_score))
        out.append(
            f"{marker}{i:>3}  {C.CYAN}{s.phase:<10}{C.RESET} {s.ident:<16} "
            f"{pad}{score_str}  {_fmt_time(s.time_s):>7}  "
            f"{rc_col}{('-' if s.returncode is None else s.returncode):>2}{C.RESET}  {note}"
        )
    out.append(f"{C.BOLD}{C.CYAN}{bar}{C.RESET}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def render_modules(traj: Trajectory) -> str:
    """The module/parent view: how each executed step maps onto an exported file
    and what it was derived from. This is what the pipeline analyzer consumes."""
    mods = [s for s in traj.steps if s.module]
    w = max((len(s.module) for s in mods), default=6)
    out = [f" {C.BOLD}{'MODULE':<{w}} {'PARENT':<{w}} {'PHASE':<10} {'SCORE':>9}{C.RESET}"]
    for s in mods:
        out.append(f" {s.module:<{w}} {C.DIM}{str(s.parent or '—'):<{w}}{C.RESET} "
                   f"{C.CYAN}{s.phase:<10}{C.RESET} {_fmt_score(s.score):>9}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Tabular overview of an MLE development trajectory.")
    ap.add_argument("json_path", help="path to the trajectory json (e.g. final_state.json)")
    ap.add_argument("--type", default="auto", choices=["auto", *sorted(PARSERS)],
                    help="trajectory format (default: detect from the file)")
    ap.add_argument("--sort", default="order", choices=("order", "time", "score"),
                    help="row ordering (default: chronological order)")
    ap.add_argument("--modules", action="store_true",
                    help="also print the exported-module / parent mapping")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    args = ap.parse_args(argv)

    if args.no_color or not sys.stdout.isatty():
        C.disable()

    try:
        with open(args.json_path) as fh:
            state = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not read {args.json_path}: {exc}", file=sys.stderr)
        return 2

    path = Path(args.json_path)
    try:
        traj = _traj.parse(state, path.resolve().parent, args.type)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not traj.steps:
        print(f"warning: no trajectory steps recognised in {args.json_path} "
              f"(type={args.type})", file=sys.stderr)
    print(render(traj, args.json_path, sort=args.sort))
    if args.modules:
        print()
        print(render_modules(traj))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
