#!/usr/bin/env python3
"""Tabular overview of an MLE development-process trajectory.

Usage:
    python tools/trajectory.py path/to/final_state.json
    python tools/trajectory.py path/to/final_state.json --type mle-star
    python tools/trajectory.py path/to/final_state.json --no-color --sort time

Extending to a new run format:
    1. Write a parser function `parse_<type>(state: dict) -> Trajectory`.
    2. Register it in the PARSERS dict at the bottom.
    That's it -- the renderer is format-agnostic and works off the Trajectory.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Callable


# --------------------------------------------------------------------------- #
# Data model shared by every parser
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    """One executed piece of code in the trajectory."""
    order: tuple          # sort key giving chronological order
    phase: str            # e.g. "Init", "Ablation", "Improve"
    ident: str            # human label for which step, e.g. "step3 plan4"
    score: float | None
    time_s: float | None
    returncode: int | None
    note: str = ""        # freeform extra (e.g. ablation summary / error hint)


@dataclass
class Trajectory:
    meta: dict = field(default_factory=dict)   # ordered key -> value for header
    steps: list[Step] = field(default_factory=list)
    lower_is_better: bool = False               # metric orientation

    def best_step(self) -> Step | None:
        scored = [s for s in self.steps if s.score is not None]
        if not scored:
            return None
        return (min if self.lower_is_better else max)(scored, key=lambda s: s.score)


# --------------------------------------------------------------------------- #
# MLE-STAR parser
# --------------------------------------------------------------------------- #
# The final_state.json is a flat dict whose keys encode the trajectory via
# numeric suffixes. We recognise the execution records (dicts carrying
# `execution_time`/`returncode`) and slot each into an ordered phase.
#
# Phase layout (chronological):
#   Init      init_code_exec_result_{sol}_{cand}
#   Merge     merger_code_exec_result_{sol}_{round}
#   ... outer refine loop, per step s ...
#     Base    train_code_exec_result_{s}_{sol}
#     Ablate  ablation_code_exec_result_{s}_{sol}
#     Improve train_code_improve_exec_result_{plan}_{s}_{sol}
#   Ensemble  ensemble_code_exec_result_{round}
#   Submit    submission_code_exec_result

_PATTERNS: list[tuple[re.Pattern, Callable]] = []


def _pat(regex: str):
    def deco(fn):
        _PATTERNS.append((re.compile(regex + r"$"), fn))
        return fn
    return deco


@_pat(r"init_code_exec_result_(\d+)_(\d+)")
def _init(sol, cand):
    return (0, 0, 0, int(cand)), "Init", f"sol{sol} cand{cand}"


@_pat(r"merger_code_exec_result_(\d+)_(\d+)")
def _merge(sol, rnd):
    return (0, 1, 0, int(rnd)), "Merge", f"sol{sol} round{rnd}"


@_pat(r"train_code_exec_result_(\d+)_(\d+)")
def _base(step, sol):
    # base/accepted solution at the start of refine step `step`
    return (1, int(step), 0, 0), "Base", f"step{step}"


@_pat(r"ablation_code_exec_result_(\d+)_(\d+)")
def _ablation(step, sol):
    return (1, int(step), 1, 0), "Ablation", f"step{step}"


@_pat(r"train_code_improve_exec_result_(\d+)_(\d+)_(\d+)")
def _improve(plan, step, sol):
    return (1, int(step), 2, int(plan)), "Improve", f"step{step} plan{plan}"


@_pat(r"ensemble_code_exec_result_(\d+)")
def _ensemble(rnd):
    return (2, 0, 0, int(rnd)), "Ensemble", f"round{rnd}"


@_pat(r"submission_code_exec_result")
def _submission():
    return (3, 0, 0, 0), "Submit", "final"


_META_KEYS = [
    ("task_name", "task"),
    ("task_type", "task type"),
    ("agent_model", "agent model"),
    ("seed", "seed"),
    ("num_solutions", "num solutions"),
    ("num_model_candidates", "model candidates"),
    ("num_top_plans", "top plans / step"),
    ("outer_loop_round", "outer rounds"),
    ("inner_loop_round", "inner rounds"),
    ("ensemble_loop_round", "ensemble rounds"),
    ("max_debug_round", "max debug rounds"),
    ("exec_timeout", "exec timeout (s)"),
    ("workspace_dir", "workspace"),
]


def parse_mle_star(state: dict) -> Trajectory:
    traj = Trajectory(lower_is_better=bool(state.get("lower", False)))

    for key, label in _META_KEYS:
        if key in state:
            traj.meta[label] = state[key]

    for key, value in state.items():
        if not (isinstance(value, dict) and
                ("execution_time" in value or "returncode" in value)):
            continue
        for pattern, fn in _PATTERNS:
            m = pattern.match(key)
            if not m:
                continue
            order, phase, ident = fn(*m.groups())
            note = ""
            if value.get("score") is None and "ablation_result" in value:
                # ablation runs report a text result rather than a score
                first = (value["ablation_result"] or "").strip().splitlines()
                note = first[0][:60] if first else "ablation study"
            elif value.get("returncode") not in (0, None):
                note = "FAILED"
            traj.steps.append(Step(
                order=order,
                phase=phase,
                ident=ident,
                score=value.get("score"),
                time_s=value.get("execution_time"),
                returncode=value.get("returncode"),
                note=note,
            ))
            break

    traj.steps.sort(key=lambda s: s.order)
    return traj


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
    ttype = traj.meta.get("task type", "")
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
            f"{rc_col}{str(s.returncode):>2}{C.RESET}  {note}"
        )
    out.append(f"{C.BOLD}{C.CYAN}{bar}{C.RESET}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Registry + CLI
# --------------------------------------------------------------------------- #
PARSERS: dict[str, Callable[[dict], Trajectory]] = {
    "mle-star": parse_mle_star,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Tabular overview of an MLE development trajectory.")
    ap.add_argument("json_path", help="path to the trajectory json (e.g. final_state.json)")
    ap.add_argument("--type", default="mle-star", choices=sorted(PARSERS),
                    help="trajectory format (default: mle-star)")
    ap.add_argument("--sort", default="order", choices=("order", "time", "score"),
                    help="row ordering (default: chronological order)")
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

    traj = PARSERS[args.type](state)
    if not traj.steps:
        print(f"warning: no trajectory steps recognised in {args.json_path} "
              f"(type={args.type})", file=sys.stderr)
    print(render(traj, args.json_path, sort=args.sort))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
