"""CLI: analyze a folder of skrub DataOps pipelines and emit an HTML report.

    python -m pipeline_analyzer --pipelines <dir> --out analysis.html

Run from the directory that contains the ``pipeline_analyzer/`` package (so
``python -m`` finds it), e.g. env_skrub/agent_run1/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .loader import load_all
from .lineage import build_lineage
from .diff import diff_dags, is_structural_noop
from .html import build_html


def _text_summary(lineage):
    print(f"{'pipeline':<14}{'score':>10}  {'Δ':>9}  change")
    print("-" * 78)
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
        print(f"{node.name:<14}{score:>10}  {dstr:>9}  {summary}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipelines", type=Path, default=Path("pipelines"),
                    help="folder of pipeline_*.py files (default: ./pipelines)")
    ap.add_argument("--results", type=Path, default=None,
                    help="results.json (default: <pipelines>/results.json)")
    ap.add_argument("--out", type=Path, default=Path("pipeline_evolution.html"))
    ap.add_argument("--title", default="Pipeline evolution")
    ap.add_argument("--unroll-choices", action="store_true",
                    help="unroll choose_from into separate branches (default: keep folded)")
    ap.add_argument("--text", action="store_true", help="print a text summary too")
    args = ap.parse_args(argv)

    pipe_dir = args.pipelines.resolve()
    if not pipe_dir.is_dir():
        ap.error(f"no such directory: {pipe_dir}")
    results = args.results or (pipe_dir / "results.json")

    print(f"Loading pipelines from {pipe_dir} …", file=sys.stderr)
    pipelines = load_all(pipe_dir, unroll_choices=args.unroll_choices)
    lineage = build_lineage(pipelines, results_path=results)

    ok = sum(1 for p in pipelines if p.ok)
    print(f"  {ok}/{len(pipelines)} extracted; "
          f"{len(pipelines) - ok} failed", file=sys.stderr)
    for p in pipelines:
        if not p.ok:
            print(f"    ! {p.name}: {(p.error or '').splitlines()[0]}", file=sys.stderr)

    html = build_html(lineage, title=args.title,
                      subtitle=pipe_dir.name, generated_note="generated")
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out.resolve()}", file=sys.stderr)

    if args.text:
        _text_summary(lineage)


if __name__ == "__main__":
    main()
