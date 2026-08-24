"""CLI: turn a plain sklearn/pandas ML script into a skrub DataOps pipeline.

    # convert (writes <dir>/skrubify/<name>.py next to the source)
    python -m skrubify path/to/train0.py --provider openai --model gpt-5

    # validate an existing pipeline -- no LLM, no dataset, no API key
    python -m skrubify --check tab_playground_dec_21/mle_star-run1/skrubify/init1.py

    # see exactly what would be sent
    python -m skrubify path/to/train0.py --print-prompt

Run from the directory containing the ``skrubify/`` package (e.g. ``tools/``), or
install the project and use the ``skrubify`` entry point.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import Result, SkrubifyConfig, build_messages, default_out_path, skrubify_file
from .llm import LLM, LLMError, PROVIDERS, load_env, missing_keys, resolve_model
from .validate import (compare_scores, parse_scores,
                       parse_scores_with_precision, run_pipeline, validate)


def _add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("sources", nargs="*", type=Path,
                    help="script(s) to skrubify")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output file (single source) or output DIRECTORY (several). "
                         "Default: <source_dir>/skrubify/<source_name>")

    g = ap.add_argument_group("model")
    g.add_argument("--provider", default=None, choices=sorted(PROVIDERS),
                   help="LLM provider; combined with --model into litellm's "
                        "'<provider>/<model>'")
    g.add_argument("--model", default=None,
                   help="model name, or a fully qualified 'provider/model'")
    g.add_argument("--temperature", type=float, default=None,
                   help="omitted by default (some models reject a non-default value)")
    g.add_argument("--max-tokens", type=int, default=None)
    g.add_argument("--retries", type=int, default=2, help="litellm num_retries")
    g.add_argument("--request-timeout", type=int, default=600)
    g.add_argument("--env-file", type=Path, default=None,
                   help="dot file with API keys (default: search .env / .env.local / "
                        ".skrubify.env in cwd upwards to the repo root, then the package)")

    g = ap.add_argument_group("prompt")
    g.add_argument("--guide", type=Path, default=None,
                   help="skrub guide markdown (default: tools/skrub_dataops_summary.md)")
    g.add_argument("--examples-dir", type=Path, default=None,
                   help="directory of NN_source.py / NN_skrub.py few-shot pairs")
    g.add_argument("--n-examples", type=int, default=None,
                   help="use only the first N few-shot pairs (0 = none)")
    g.add_argument("--instructions", default=None,
                   help="extra task-specific instructions appended to the prompt")
    g.add_argument("--instructions-file", type=Path, default=None)

    g = ap.add_argument_group("validation")
    g.add_argument("--max-repairs", type=int, default=2,
                   help="LLM repair rounds after a failed validation (default 2)")
    g.add_argument("--no-validate", action="store_true",
                   help="skip the plan build (static checks still run)")
    g.add_argument("--strict", action="store_true",
                   help="treat warnings as errors")
    g.add_argument("--python", default=None,
                   help="interpreter used to build the plan (default: this one)")
    g.add_argument("--keep-attempts", action="store_true",
                   help="also write each attempt to <stem>.attemptN.py")
    g.add_argument("--build-timeout", type=int, default=120,
                   help="seconds allowed for one plan build (default 120)")

    g = ap.add_argument_group("run (needs the dataset)")
    g.add_argument("--run-in", type=Path, default=None, metavar="DIR",
                   help="after validating, EXECUTE the pipeline with this working "
                        "directory (the one its relative paths resolve against, e.g. "
                        "the dir containing ./input) and report the score it prints. "
                        "The only check that catches scoring-time failures.")
    g.add_argument("--compare-source", action="store_true",
                   help="with --run-in, also run the ORIGINAL script and report both "
                        "scores and their difference")
    g.add_argument("--run-timeout", type=int, default=1800)

    g = ap.add_argument_group("modes")
    g.add_argument("--check", action="store_true",
                   help="validate the given file(s) as skrub pipelines and exit "
                        "(no LLM call)")
    g.add_argument("--print-prompt", action="store_true",
                   help="print the assembled prompt and exit (no LLM call)")
    g.add_argument("--json-report", type=Path, default=None,
                   help="write a machine-readable run report")
    g.add_argument("-q", "--quiet", action="store_true")


def _run_and_compare(pipeline: Path, source: Path | None, args) -> int:
    """Execute the pipeline (and optionally the original) and report the scores."""
    print(f"  running {pipeline.name} in {args.run_in} …", file=sys.stderr, flush=True)
    ok, score, detail = run_pipeline(pipeline, args.run_in, python=args.python,
                                     timeout=args.run_timeout)
    if not ok:
        print(f"  RUN FAILED: {detail}")
        return 1
    new_scores = parse_scores(detail) or ([score] if score is not None else [])
    if len(new_scores) > 1:
        print(f"  skrubified scores ({len(new_scores)} variants): "
              f"{[round(v, 6) for v in new_scores]}")
    else:
        print(f"  skrubified score: {score!r}")
    if not (args.compare_source and source):
        return 0
    print(f"  running {source.name} (original) in {args.run_in} …",
          file=sys.stderr, flush=True)
    ok_s, score_s, detail_s = run_pipeline(source, args.run_in, python=args.python,
                                           timeout=args.run_timeout)
    old_scores, tol = parse_scores_with_precision(detail_s)
    if old_scores and new_scores:
        # Compare the FULL score lists: a multi-variant original prints one score
        # per experiment and the fused plan one per grid row.
        if len(old_scores) > 1 or len(new_scores) > 1:
            print(f"  original scores ({len(old_scores)}): "
                  f"{[round(v, 6) for v in old_scores]}")
            print(f"  comparison:       "
                  f"{compare_scores(new_scores, old_scores, tol=tol)}")
            return 0
    if not ok_s:
        # A multi-variant source (an ablation study) prints several scores and no
        # single "Final Validation Performance" line, so there is nothing to diff
        # automatically -- show its output and let the reader compare it with the
        # grid the fused plan printed.
        print("  no single score parsed from the original; its output was:")
        print("\n".join(f"    | {line}" for line in detail_s.splitlines()[-25:]))
        return 0
    delta = score - score_s
    verdict = ("identical" if score == score_s else
               "close" if abs(delta) < 1e-3 else "DIFFERENT")
    print(f"  original score:   {score_s!r}")
    print(f"  delta:            {delta:+.6g}  ({verdict})")
    return 0


def _check(paths: list[Path], args) -> int:
    rc = 0
    for path in paths:
        v = validate(path, python=args.python, strict=args.strict,
                     build=not args.no_validate, timeout=args.build_timeout)
        flag = "OK  " if v.ok else "FAIL"
        print(f"{flag} {path}: {v.summary()}")
        text = v.feedback()
        if text:
            print("\n".join(f"     {line}" for line in text.splitlines()))
        if v.ok and args.run_in:
            rc |= _run_and_compare(path, None, args)
        rc |= 0 if v.ok else 1
    return rc


def _resolve_out(args, source: Path, n_sources: int) -> Path:
    if args.out is None:
        return default_out_path(source)
    if n_sources > 1 or args.out.is_dir() or not args.out.suffix:
        return args.out / source.name
    return args.out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="skrubify", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_args(ap)
    args = ap.parse_args(argv)

    if not args.sources:
        ap.error("give at least one source script (or --check FILE)")
    missing = [p for p in args.sources if not p.is_file()]
    if missing:
        ap.error(f"no such file(s): {', '.join(map(str, missing))}")

    if args.check:
        return _check(args.sources, args)

    instructions = args.instructions
    if args.instructions_file:
        instructions = ((instructions + "\n\n") if instructions else "") + \
                       args.instructions_file.read_text()

    cfg = SkrubifyConfig(
        guide_path=args.guide, examples_dir=args.examples_dir,
        n_examples=args.n_examples, extra_instructions=instructions,
        max_repairs=args.max_repairs, validate_build=not args.no_validate,
        strict=args.strict, python=args.python, timeout=args.build_timeout,
        keep_attempts=args.keep_attempts, verbose=not args.quiet,
    )

    if args.print_prompt:
        for src in args.sources:
            for m in build_messages(src, cfg):
                print(f"===== {m['role']} =====\n{m['content']}\n")
        return 0

    loaded = load_env(args.env_file)
    try:
        model = resolve_model(args.model, args.provider)
    except ValueError as e:
        ap.error(str(e))
    lacking = missing_keys(model)
    if lacking:
        print(f"! missing API key(s) for {model}: {', '.join(lacking)}\n"
              f"  looked in: {', '.join(map(str, loaded)) or 'no dot file found'}\n"
              f"  put them in a .env file (KEY=value) or export them.",
              file=sys.stderr)
        return 2
    llm = LLM(model=model, temperature=args.temperature, max_tokens=args.max_tokens,
              num_retries=args.retries, timeout=args.request_timeout)
    cfg.llm = llm

    results: list[Result] = []
    rc = 0
    for src in args.sources:
        out = _resolve_out(args, src, len(args.sources))
        if not args.quiet:
            print(f"skrubify {src} -> {out}  [{model}]", file=sys.stderr)
        try:
            res = skrubify_file(src, out, cfg=cfg)
        except LLMError as exc:
            # Auth/credit/quota failures are global, so stop rather than repeat the
            # same error once per remaining file. Nothing was written for `src`.
            remaining = args.sources[args.sources.index(src):]
            print(f"! API call failed: {exc}", file=sys.stderr)
            print(f"! stopped: {len(remaining)} file(s) not converted "
                  f"({', '.join(p.name for p in remaining[:5])}"
                  f"{', …' if len(remaining) > 5 else ''}). "
                  "Existing outputs were left untouched.", file=sys.stderr)
            return 3
        results.append(res)
        status = "ok" if res.ok else "NEEDS REVIEW"
        print(f"{status}: {out}  ({len(res.attempts)} attempt(s))")
        rc |= 0 if res.ok else 1
        if args.run_in and res.ok:
            rc |= _run_and_compare(out, src, args)

    if not args.quiet:
        s = llm.stats()
        print(f"[{s['model']}] {s['calls']} call(s), "
              f"{s['prompt_tokens']}+{s['completion_tokens']} tokens, "
              f"${s['cost_usd']:.4f}", file=sys.stderr)
    if args.json_report:
        args.json_report.write_text(json.dumps(
            {"model": model, "results": [r.as_dict() for r in results]}, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
