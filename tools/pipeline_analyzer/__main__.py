"""CLI: analyze a folder of skrub DataOps pipelines and emit an HTML report.

Two ways to get the lineage:

    # pipelines annotate their own PARENT (see skrub_dataops_guide.md)
    python -m pipeline_analyzer --pipelines <dir> --out analysis.html

    # an agent trajectory supplies parents + scores, the folder holds the
    # (hand-)skrubified rewrite of each step's code
    python -m pipeline_analyzer --trajectory final_state.json \
        --pipelines skrubify_openai --pipelines skrubify_openai/ensemble

``--pipelines`` may be repeated; a trajectory step is matched to the first folder
that holds ``<module>.py``. Steps that were never skrubified are reported and
skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .loader import load_all, resolve
from .lineage import (build_lineage, build_lineage_from_trajectory,
                      fold_translation_variants)
from .diff import diff_dags, is_structural_noop
from .html import build_html
from .trajectory import PARSERS, parse as parse_trajectory


def _text_summary(lineage):
    phased = any(n.phase for n in lineage.nodes.values())
    w = max((len(n) for n in lineage.nodes), default=8) + 2
    head = f"{'pipeline':<{w}}{'phase':<10}" if phased else f"{'pipeline':<{w}}"
    print(f"{head}{'score':>10}  {'Δ':>9}  change")
    print("-" * (len(head) + 32))
    for node in lineage.ordered():
        p = node.pipeline
        score = f"{node.score:.5f}" if node.score is not None else "—"
        delta = lineage.delta_score(node.name)
        dstr = f"{delta:+.4f}" if delta is not None else "—"
        if not p.ok:
            summary = f"FAILED: {(p.error or '').splitlines()[0]}"
        else:
            parent = lineage.nodes.get(node.parent) if node.parent else None
            pdag = parent.pipeline.dag if (parent and parent.pipeline.ok) else None
            diff = diff_dags(pdag, p.dag)
            if pdag is None:
                summary = f"root · {len(p.dag.nodes)} ops"
            else:
                bits = []
                nf, nr = len(diff.frontier), len(diff.removed)
                if nf or nr:
                    bits.append(f"+{nf}/−{nr} ops")
                ne = len(diff.estimator_deltas)
                if ne:
                    bits.append(f"{ne} estimator change(s)")
                if is_structural_noop(diff) and ne:
                    bits.append("(hyperparameters only)")
                summary = ", ".join(bits) or "no change"
        label = (f"{node.name:<{w}}{(node.phase or ''):<10}" if phased
                 else f"{node.name:<{w}}")
        print(f"{label}{score:>10}  {dstr:>9}  {summary}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipelines", type=Path, action="append", default=None,
                    metavar="DIR", help="folder of pipeline files; repeatable "
                                        "(default: ./pipelines)")
    ap.add_argument("--trajectory", type=Path, default=None,
                    help="agent trajectory json (e.g. final_state.json) supplying "
                         "lineage + scores instead of PARENT/results.json")
    ap.add_argument("--trajectory-type", default="auto",
                    choices=["auto", *sorted(PARSERS)],
                    help="trajectory format (default: detect from the file)")
    ap.add_argument("--results", type=Path, default=None,
                    help="results.json (default: <pipelines>/results.json); "
                         "ignored with --trajectory")
    ap.add_argument("--out", type=Path, default=Path("pipeline_evolution.html"))
    ap.add_argument("--title", default=None)
    ap.add_argument("--unroll-choices", action="store_true",
                    help="unroll choose_from into separate branches (default: keep folded)")
    ap.add_argument("--fold-identical-code", action="store_true",
                    help="with --trajectory: steps whose original code is "
                         "byte-identical share one skrubified DAG, so a step that "
                         "changed nothing diffs as unchanged instead of showing "
                         "the rewrite-to-rewrite translation noise")
    ap.add_argument("--text", action="store_true", help="print a text summary too")
    args = ap.parse_args(argv)

    pipe_dirs = [d.resolve() for d in (args.pipelines or [Path("pipelines")])]
    for d in pipe_dirs:
        if not d.is_dir():
            ap.error(f"no such directory: {d}")

    traj = None
    if args.trajectory:
        try:
            state = json.loads(args.trajectory.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            ap.error(f"could not read {args.trajectory}: {exc}")
        try:
            traj = parse_trajectory(state, args.trajectory.resolve().parent,
                                    args.trajectory_type)
        except ValueError as exc:
            ap.error(str(exc))
        if not traj.steps:
            ap.error(f"no steps recognised in {args.trajectory} "
                     f"(type={args.trajectory_type})")

    where = ", ".join(str(d) for d in pipe_dirs)
    print(f"Loading pipelines from {where} …", file=sys.stderr)
    if traj is not None:
        names = list(traj.by_module())
        found = {n for n, _ in resolve(pipe_dirs, names)}
        missing = [n for n in names if n not in found]
        print(f"  trajectory: {len(traj.steps)} steps, "
              f"{len(found)} skrubified, {len(missing)} without a file",
              file=sys.stderr)
        if missing:
            print(f"    not skrubified: {' '.join(missing)}", file=sys.stderr)
        pipelines = load_all(pipe_dirs, names=names, unroll_choices=args.unroll_choices)
        if args.fold_identical_code:
            folded = fold_translation_variants(pipelines, traj)
            print(f"  folded {len(folded)} code-identical step(s) onto their first "
                  f"skrubified rewrite", file=sys.stderr)
        lineage = build_lineage_from_trajectory(pipelines, traj)
    else:
        pipelines = load_all(pipe_dirs, unroll_choices=args.unroll_choices)
        results = args.results or (pipe_dirs[0] / "results.json")
        lineage = build_lineage(pipelines, results_path=results)

    if not pipelines:
        hint = ""
        sibs = [d.name for d in pipe_dirs[0].parent.glob("skrubify*") if d.is_dir()]
        if pipe_dirs[0].name == "pipelines" and sibs:
            hint = (f"\n  a run's pipelines/ folder holds the agent's ORIGINAL scripts; "
                    f"point --pipelines at the skrubified plans instead "
                    f"({', '.join(sibs)})")
        ap.error(f"no pipelines loaded from {where}{hint}")

    ok = sum(1 for p in pipelines if p.ok)
    print(f"  {ok}/{len(pipelines)} extracted; "
          f"{len(pipelines) - ok} failed", file=sys.stderr)
    for p in pipelines:
        if not p.ok:
            print(f"    ! {p.name}: {(p.error or '').splitlines()[0]}", file=sys.stderr)

    title = args.title or (
        f"{traj.meta.get('task', 'run')} — pipeline evolution" if traj
        else "Pipeline evolution")
    subtitle = pipe_dirs[0].name
    if traj is not None:
        bits = [b for b in (traj.meta.get("format"), traj.meta.get("agent model"),
                            subtitle) if b]
        subtitle = " · ".join(bits)
    html = build_html(lineage, title=title, subtitle=subtitle,
                      generated_note="generated")
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out.resolve()}", file=sys.stderr)

    if args.text:
        _text_summary(lineage)


if __name__ == "__main__":
    main()
