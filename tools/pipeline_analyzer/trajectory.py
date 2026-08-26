"""Read an agent run's *trajectory* (its final state dump) into ordered steps.

One :class:`Step` per executed piece of code, carrying the score/time/returncode
the harness recorded, plus two things that let the pipeline analyzer work off a
trajectory instead of ``PARENT`` annotations in the pipeline files:

* ``module`` -- the file stem the run's code was exported to (``train3_improve1``),
  so a folder of hand-skrubified pipelines can be matched back to the trajectory.
* ``parent`` -- the module this one was derived from, reconstructed from the
  search structure of the agent (see :func:`parse_mle_star`).

Stdlib only, on purpose: ``tools/trajectory.py`` loads this module directly (no
package import, hence no skrub/stratum/graphviz) just to print a table.

Extending to a new run format:
    1. Write ``parse_<type>(state: dict, base: Path | None) -> Trajectory``
       (``base`` is the folder the state file lives in, for reading code off disk).
    2. Register it in :data:`PARSERS`, and give :func:`detect_type` a way to
       recognise it.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
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
    key: str = ""         # the state key this step was read from
    module: str | None = None   # exported file stem, e.g. "train3_improve1"
    parent: str | None = None   # module this one was derived from
    desc: str | None = None     # the agent's rationale (plan / ablation summary)
    code_sig: str | None = None  # hash of the *original* code, for identity checks


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

    def by_module(self) -> dict[str, Step]:
        """Steps that were exported to a file, keyed by module stem."""
        return {s.module: s for s in self.steps if s.module}


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
#
# Each handler also returns the file stem the run's exporter writes that code to
# (``module``); the code itself lives under the same key minus "_exec_result".

_PATTERNS: list[tuple[re.Pattern, Callable]] = []


def _pat(regex: str):
    def deco(fn):
        _PATTERNS.append((re.compile(regex + r"$"), fn))
        return fn
    return deco


def _sfx(sol: str) -> str:
    """Solution suffix. The exporter drops it for the single-solution case (the
    only one seen so far); keep it explicit when a run carries several."""
    return "" if sol == "1" else f"_sol{sol}"


@_pat(r"init_code_exec_result_(\d+)_(\d+)")
def _init(sol, cand):
    return (0, 0, 0, int(cand)), "Init", f"sol{sol} cand{cand}", f"init_code_{cand}{_sfx(sol)}"


@_pat(r"merger_code_exec_result_(\d+)_(\d+)")
def _merge(sol, rnd):
    return (0, 1, 0, int(rnd)), "Merge", f"sol{sol} round{rnd}", f"train0_{rnd}{_sfx(sol)}"


@_pat(r"train_code_exec_result_(\d+)_(\d+)")
def _base(step, sol):
    # base/accepted solution at the start of refine step `step`
    return (1, int(step), 0, 0), "Base", f"step{step}", f"train{step}{_sfx(sol)}"


@_pat(r"ablation_code_exec_result_(\d+)_(\d+)")
def _ablation(step, sol):
    return (1, int(step), 1, 0), "Ablation", f"step{step}", f"ablation_{step}{_sfx(sol)}"


@_pat(r"train_code_improve_exec_result_(\d+)_(\d+)_(\d+)")
def _improve(plan, step, sol):
    return ((1, int(step), 2, int(plan)), "Improve", f"step{step} plan{plan}",
            f"train{step}_improve{plan}{_sfx(sol)}")


@_pat(r"ensemble_code_exec_result_(\d+)")
def _ensemble(rnd):
    return (2, 0, 0, int(rnd)), "Ensemble", f"round{rnd}", f"ensemble{rnd}"


@_pat(r"submission_code_exec_result")
def _submission():
    return (3, 0, 0, 0), "Submit", "final", "final_solution"


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


def _code_key(exec_key: str) -> str:
    """State key holding the code an ``*_exec_result_*`` record was produced from."""
    return exec_key.replace("_exec_result", "")


def _code_sig(code) -> str | None:
    """Hash of a step's source. Equal hashes mean the run re-used code verbatim
    (an accepted variant copied into the next base, or a step that changed
    nothing), which is how parents are identified and how a hand-skrubified
    folder can be told apart from genuinely new code."""
    if not isinstance(code, str) or not code.strip():
        return None
    return hashlib.sha1(code.strip().encode()).hexdigest()[:16]


def _first_line(text, limit=60):
    lines = (text or "").strip().splitlines()
    return lines[0][:limit] if lines else ""


def _describe(state: dict, phase: str, groups: tuple) -> str | None:
    """The agent's own rationale for this step, where the state records one."""
    if phase == "Improve":
        plan, step, sol = groups
        plans = state.get(f"refine_plans_{step}_{sol}") or []
        idx = int(plan)
        if idx < len(plans):
            return str(plans[idx])
    elif phase == "Ablation":
        step, sol = groups
        return state.get(f"ablation_summary_{step}_{sol}") or None
    elif phase == "Ensemble":
        plans = state.get("ensemble_plans") or []
        idx = int(groups[0])
        if idx < len(plans):
            return str(plans[idx])
    return None


def parse_mle_star(state: dict, base: Path | None = None) -> Trajectory:
    """MLE-STAR: a flat state dump whose keys encode the search (see above). The
    code lives in the state itself, so ``base`` is unused."""
    traj = Trajectory(lower_is_better=bool(state.get("lower", False)))

    for key, label in _META_KEYS:
        if key in state:
            traj.meta[label] = state[key]

    recs: list[tuple[Step, tuple]] = []      # (step, regex groups)
    for key, value in state.items():
        if not (isinstance(value, dict) and
                ("execution_time" in value or "returncode" in value)):
            continue
        for pattern, fn in _PATTERNS:
            m = pattern.match(key)
            if not m:
                continue
            order, phase, ident, module = fn(*m.groups())
            note = ""
            if value.get("score") is None and "ablation_result" in value:
                # ablation runs report a text result rather than a score
                note = _first_line(value["ablation_result"]) or "ablation study"
            elif value.get("returncode") not in (0, None):
                note = "FAILED"
            step = Step(
                order=order,
                phase=phase,
                ident=ident,
                score=value.get("score"),
                time_s=value.get("execution_time"),
                returncode=value.get("returncode"),
                note=note,
                key=key,
                module=module,
                desc=_describe(state, phase, m.groups()),
                code_sig=_code_sig(state.get(_code_key(key))),
            )
            recs.append((step, m.groups()))
            traj.steps.append(step)
            break

    _link_parents(state, recs, traj.lower_is_better)
    traj.steps.sort(key=lambda s: s.order)
    return traj


def _link_parents(state: dict, recs: list[tuple[Step, tuple]], lower: bool) -> None:
    """Reconstruct "derived from" edges from MLE-STAR's search structure.

    The state records no parent pointers, but the algorithm's shape gives them:
    candidates are independent roots; the merger chain starts from the winning
    candidate; each refine step's base is the previous step's *accepted* variant;
    ablations and plan variants branch off their step's base.

    Where the winner is what matters (which candidate the merger started from,
    which plan variant was accepted), we identify it by **code identity** -- the
    accepted variant's code is copied verbatim into the next base, so a string
    match is exact -- and fall back to the best score when no code matches (e.g.
    an accepted variant that was subsequently debugged).
    """
    code = {s.module: (state.get(_code_key(s.key)) or "").strip() for s, _ in recs}

    def by_phase(phase):
        return sorted((r for r in recs if r[0].phase == phase), key=lambda r: r[0].order)

    def best(cands):
        scored = [c for c in cands if c[0].score is not None]
        if not scored:
            return cands[0][0].module if cands else None
        return (min if lower else max)(scored, key=lambda c: c[0].score)[0].module

    def same_code(module, cands):
        """The candidate whose code is byte-identical to ``module``'s."""
        target = code.get(module) or ""
        for c in cands:
            if target and code.get(c[0].module) == target:
                return c[0].module
        return None

    inits, mergers = by_phase("Init"), by_phase("Merge")
    bases = {(r[1][1], int(r[1][0])): r for r in by_phase("Base")}   # (sol, step) -> rec
    improves: dict[tuple, list] = {}
    for r in by_phase("Improve"):
        improves.setdefault((r[1][2], int(r[1][1])), []).append(r)

    per_sol = lambda seq, sol, i: [r for r in seq if r[1][i] == sol]   # noqa: E731

    for r in mergers:
        sol, rnd = r[1][0], int(r[1][1])
        if rnd == 0:
            cands = per_sol(inits, sol, 0)
            r[0].parent = same_code(r[0].module, cands) or best(cands)
        else:
            prev = [m for m in mergers if m[1] == (sol, str(rnd - 1))]
            r[0].parent = prev[0][0].module if prev else None

    for (sol, step), r in sorted(bases.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if step == 0:
            chain = per_sol(mergers, sol, 0)
            r[0].parent = (same_code(r[0].module, chain)
                           or (chain[-1][0].module if chain else None)
                           or best(per_sol(inits, sol, 0)))
        else:
            prev_variants = improves.get((sol, step - 1), [])
            prev_base = bases.get((sol, step - 1))
            r[0].parent = (same_code(r[0].module, prev_variants)
                           or (prev_base[0].module if prev_base else None))

    for r in by_phase("Ablation"):
        base = bases.get((r[1][1], int(r[1][0])))
        r[0].parent = base[0].module if base else None
    for r in by_phase("Improve"):
        base = bases.get((r[1][2], int(r[1][1])))
        r[0].parent = base[0].module if base else None

    # Ensembling and submission consume the finished solutions; the last accepted
    # base is the closest thing to a single parent (best one, if several).
    last_bases = {}
    for (sol, step), r in bases.items():
        if step > last_bases.get(sol, (-1, None))[0]:
            last_bases[sol] = (step, r)
    tail = best([r for _, r in last_bases.values()]) if last_bases else None

    ens = by_phase("Ensemble")
    for i, r in enumerate(ens):
        r[0].parent = ens[i - 1][0].module if i else tail
    for r in by_phase("Submit"):
        r[0].parent = ens[-1][0].module if ens else tail


# --------------------------------------------------------------------------- #
# mlevolve parser
# --------------------------------------------------------------------------- #
# journal_slim.json is a proper search tree: ``nodes`` (one per attempted
# solution) plus ``node2parent``. Unlike MLE-STAR nothing has to be inferred --
# parents are recorded, and each node points at the file its code was written to
# (``code_file``, relative to the run folder). Buggy nodes can lack a code file;
# they still become steps (they are part of the run) but hold no module, so a
# child's parent is lifted to the nearest ancestor that does have one.

_STAGE_PHASE = {"root": "Root", "draft": "Draft", "debug": "Debug",
                "improve": "Improve", "evolution": "Evolution"}


def _plan_text(plan) -> str | None:
    """A node's rationale. Some plans are a JSON object (``reason`` + fields),
    some are plain prose."""
    if not isinstance(plan, str) or not plan.strip() or plan.strip() == "(root)":
        return None
    text = plan.strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(obj, dict):
            parts = [str(obj[k]) for k in ("reason", "plan", "description", "summary")
                     if obj.get(k)]
            return "\n\n".join(parts) or text
    return text


def parse_mlevolve(state: dict, base: Path | None = None) -> Trajectory:
    """mlevolve / AIDE-style journal: an explicit tree of solution nodes."""
    nodes = [n for n in (state.get("nodes") or []) if isinstance(n, dict)]
    parents = state.get("node2parent") or {}

    maximize = next((n["metric"].get("maximize") for n in nodes
                     if isinstance(n.get("metric"), dict)
                     and n["metric"].get("maximize") is not None), None)
    traj = Trajectory(lower_is_better=(maximize is False))

    module = {n.get("id"): (Path(n["code_file"]).stem if n.get("code_file") else None)
              for n in nodes}
    coded = sum(1 for m in module.values() if m)
    if base is not None:
        traj.meta["task"] = base.name
    traj.meta["nodes"] = f"{len(nodes)} ({coded} with code)"
    stages = {}
    for n in nodes:
        stages[n.get("stage") or "?"] = stages.get(n.get("stage") or "?", 0) + 1
    traj.meta["stages"] = ", ".join(f"{k} {v}" for k, v in sorted(stages.items()))

    def coded_ancestor(node_id):
        """Nearest ancestor that has a code file -- the parent a pipeline diff can
        actually be taken against."""
        seen, cur = set(), parents.get(node_id)
        while cur and cur not in seen:
            if module.get(cur):
                return module[cur]
            seen.add(cur)
            cur = parents.get(cur)
        return None

    for n in sorted(nodes, key=lambda n: (n.get("step") or 0, n.get("ctime") or 0)):
        stage = n.get("stage") or "?"
        metric = n.get("metric") if isinstance(n.get("metric"), dict) else {}
        buggy = n.get("is_buggy")
        note = n.get("exc_type") or ("buggy" if buggy else "")
        mod = module.get(n.get("id"))
        traj.steps.append(Step(
            order=(int(n.get("step") or 0),),
            phase=_STAGE_PHASE.get(stage, stage.capitalize()),
            ident=f"step{n.get('step')}",
            score=metric.get("value"),
            time_s=n.get("exec_time"),
            returncode=None if buggy is None else (1 if buggy else 0),
            note=note,
            key=str(n.get("id") or ""),
            module=mod,
            parent=coded_ancestor(n.get("id")),
            desc=_plan_text(n.get("plan")) or n.get("code_summary"),
            code_sig=_file_sig(base / n["code_file"]) if (base and n.get("code_file")) else None,
        ))
    return traj


def _file_sig(path: Path) -> str | None:
    """Hash of an on-disk source file (mlevolve keeps code beside the journal)."""
    try:
        return _code_sig(path.read_text())
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
PARSERS: dict[str, Callable[..., Trajectory]] = {
    "mle-star": parse_mle_star,
    "mlevolve": parse_mlevolve,
}


def detect_type(state: dict) -> str | None:
    """Guess the format from the shape of the state dump."""
    if isinstance(state.get("nodes"), list) and "node2parent" in state:
        return "mlevolve"
    if any(k.startswith("init_code_exec_result_") for k in state):
        return "mle-star"
    return None


def parse(state: dict, base: Path | None = None, kind: str = "auto") -> Trajectory:
    """Parse ``state``, detecting the format when ``kind`` is ``"auto"``."""
    if kind == "auto":
        kind = detect_type(state)
        if kind is None:
            raise ValueError("could not detect the trajectory format; pass one "
                             f"of {sorted(PARSERS)}")
    traj = PARSERS[kind](state, base)
    traj.meta["format"] = kind
    return traj
