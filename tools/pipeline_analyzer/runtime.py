"""Collect per-pipeline runtime statistics by executing pipelines under stratum.

    python -m pipeline_analyzer.runtime --pipelines skrubify_openai \
        --pipelines skrubify_openai/ensemble --run-in ../ --sample-rows 200000

Executing a pipeline needs the real dataset and is expensive, so this is a
separate command from the report: it writes a JSON store next to the pipelines
folder (``runtime_stats_<folder>.json``) and the analyzer only *reads* that file.
Re-running skips every pipeline already measured with the same code and sample
size, so a sweep can be filled in over several sessions, and the store is
rewritten after each pipeline so an interrupted sweep keeps what it collected.

Each pipeline runs in its own process (see ``_measure.py``) with ``--run-in`` as
the working directory, under a timeout, and its peak RSS is recorded.

This module is deliberately stdlib-only -- the heavy imports (skrub, stratum)
happen in the child, so ``--only``/listing/reporting stays instant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

STORE_VERSION = 1

#: "SomeError: detail" at the start of a line -- the last line of a traceback.
_EXC_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception|Exit|Interrupt)\b")


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
def default_store_path(pipe_dir: Path) -> Path:
    """``…/mle_star/skrubify_openai`` -> ``…/mle_star/runtime_stats_skrubify_openai.json``
    (beside the run's report, not inside the pipelines folder)."""
    return pipe_dir.parent / f"runtime_stats_{pipe_dir.name}.json"


def load_store(path: Path) -> dict:
    """Read a store, or an empty one when the file is absent/unreadable."""
    try:
        store = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {"version": STORE_VERSION, "meta": {}, "pipelines": {}}
    store.setdefault("pipelines", {})
    store.setdefault("meta", {})
    return store


def save_store(path: Path, store: dict) -> None:
    """Write the store atomically, so an interrupt cannot truncate it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(store, fh, indent=1, sort_keys=False)
        # mkstemp is 0600; the store is a committed artifact, so give it the
        # ordinary umask-derived mode instead of owner-only.
        os.chmod(tmp, 0o666 & ~_umask())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _umask() -> int:
    current = os.umask(0o022)
    os.umask(current)
    return current


def code_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def is_fresh(entry: dict, path: Path, sample_rows: int | None,
             mem_mode: str = "process", stratum_commit: str | None = None) -> bool:
    """Whether a stored measurement still describes this file and this sweep.

    An entry measured without memory sampling is stale for a sweep that wants it
    (and vice versa), so turning tracking on backfills the store instead of
    needing ``--force``.

    ``stratum_commit`` makes a measurement stale when the executor's stratum
    build changed, because the numbers are not comparable across builds: the
    logical optimizer decides the DAG, so an upgrade changes both the op mix a
    pipeline is credited with and how long it takes. ``stratum.__version__`` is a
    static dev string that cannot see this -- the pin moved from 834dc029 to
    8b7f7ba3 with both reporting ``0.0.0.dev2``, which is why the commit is what
    is compared. An entry from before the commit was recorded carries no commit,
    so it is stale as soon as the current executor has one.
    """
    if (entry.get("status") != "ok"
            or entry.get("code_sha1") != code_sha1(path)
            or entry.get("sample_rows") != sample_rows):
        return False
    if stratum_commit and entry.get("stratum_commit") != stratum_commit:
        return False
    return bool(entry.get("memory")) == (mem_mode != "off")


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
def run_one(path: Path, *, run_in: Path, python: str, timeout: float,
            sample_rows: int | None, stats: bool, mem_mode: str = "process",
            mem_interval: float = 0.1, mem_csv: Path | None = None) -> dict:
    """Measure one pipeline in a subprocess. Never raises."""
    fd, out_json = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    cmd = [python, "-m", "pipeline_analyzer._measure", str(Path(path).resolve()),
           out_json]
    if sample_rows:
        cmd += ["--sample-rows", str(sample_rows)]
    if not stats:
        cmd.append("--no-stats")
    cmd += ["--mem-mode", mem_mode, "--mem-interval", str(mem_interval)]
    if mem_csv:
        mem_csv.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--mem-csv", str(mem_csv)]

    env = dict(os.environ)
    # The child imports pipeline_analyzer._measure; give it this process's path.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in sys.path if p and p not in (".",))

    t0 = time.perf_counter()
    # New session so a timeout can kill the joblib/OpenMP workers an estimator
    # spawned, not just the interpreter that started them.
    proc = subprocess.Popen(cmd, cwd=str(run_in), env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    timed_out = False
    try:
        log, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        log, _ = proc.communicate()
    elapsed = time.perf_counter() - t0

    entry: dict = {}
    try:
        entry = json.loads(Path(out_json).read_text())
    except (OSError, json.JSONDecodeError):
        pass
    finally:
        Path(out_json).unlink(missing_ok=True)

    if timed_out:
        entry = {"status": "timeout",
                 "error": f"killed after {timeout:.0f}s", "sample_rows": sample_rows}
    elif not entry:
        entry = {"status": "error", "error": _failure_reason(log, proc.returncode),
                 "sample_rows": sample_rows}
    if entry.get("status") != "ok":
        entry["returncode"] = proc.returncode
        # Keep both ends of the log: a side-car process writing to the same pipe
        # unbuffered can land its traceback *before* the workload's block-buffered
        # output, so a tail alone can miss the actual failure.
        entry["log"] = _log_excerpt(log)

    entry["elapsed_s"] = round(elapsed, 3)
    entry["code_sha1"] = code_sha1(path)
    entry["measured_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return entry


def _failure_reason(log: str | None, returncode: int | None) -> str:
    """The most informative one-liner available for a child that wrote no result.

    Prefers the last exception line anywhere in the log; the child's *last* line
    is often the workload's own success output, which reads as a passing run.
    """
    lines = [ln.strip() for ln in (log or "").splitlines() if ln.strip()]
    exceptions = [ln for ln in lines if _EXC_RE.match(ln)]
    if exceptions:
        return exceptions[-1]
    if returncode and returncode < 0:
        return f"killed by signal {-returncode}"
    if lines:
        return f"exit {returncode}: {lines[-1]}"
    return f"exit {returncode}: no output"


def _log_excerpt(log: str | None, head: int = 2000, tail: int = 1500) -> str:
    log = log or ""
    if len(log) <= head + tail:
        return log
    return f"{log[:head]}\n[... {len(log) - head - tail} chars omitted ...]\n{log[-tail:]}"


def discover(dirs: list[Path]) -> list[tuple[str, Path]]:
    """Every pipeline file in ``dirs``, first folder wins on a name clash."""
    out, seen = [], set()
    for d in dirs:
        for f in sorted(d.glob("*.py")):
            if f.stem.startswith("_") or f.stem in seen:
                continue
            seen.add(f.stem)
            out.append((f.stem, f))
    return out


def _versions(python: str) -> dict:
    """Version stamp for the store, read from the interpreter doing the runs."""
    # stratum.__version__ is a static dev string, so pull the installed commit
    # from the wheel's direct_url metadata: runtime numbers are only comparable
    # against a known executor.
    code = ("import json,sys,stratum,skrub,sklearn;"
            "import importlib.metadata as md;"
            "rev=(json.loads(md.distribution('stratum-ai').read_text("
            "'direct_url.json') or '{}').get('vcs_info') or {}).get('commit_id');"
            "print(json.dumps({'stratum': stratum.__version__,"
            "'stratum_commit': rev,"
            "'skrub': skrub.__version__, 'sklearn': sklearn.__version__,"
            "'python': sys.version.split()[0]}))")
    try:
        return json.loads(subprocess.run([python, "-c", code], text=True,
                                         capture_output=True, timeout=120).stdout)
    except Exception:  # noqa: BLE001 - a version stamp is not worth failing over
        return {}


def _fmt(entry: dict) -> str:
    if entry.get("status") != "ok":
        return f"{entry.get('status')}: {(entry.get('error') or '')[:70]}"
    score = entry.get("best_score")
    rows = entry.get("sample_rows")
    mem = entry.get("memory") or {}
    peak = mem.get("peak_mb") or entry.get("max_rss_mb") or 0
    bits = [f"{entry['wall_s']:.1f}s", f"{peak:.0f}MB peak",
            f"{rows:,} rows" if rows else "full data",
            f"{len(entry.get('ops') or ())} op types"]
    if score is not None:
        bits.append(f"score {score:.5f}")
    return "ok · " + " · ".join(bits)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="pipeline_analyzer.runtime",
        description="Execute skrub pipelines under stratum and store runtime stats.")
    ap.add_argument("--pipelines", type=Path, action="append", default=None,
                    metavar="DIR", help="folder of pipeline files; repeatable "
                                        "(default: ./pipelines)")
    ap.add_argument("--run-in", type=Path, default=None, metavar="DIR",
                    help="working directory for the pipelines -- the one their "
                         "relative paths resolve against, i.e. the folder holding "
                         "input/ (default: search upwards from --pipelines)")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON store (default: runtime_stats_<folder>.json beside "
                         "the first --pipelines folder)")
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME",
                    help="measure only these pipelines")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N pipelines actually measured")
    ap.add_argument("--sample-rows", type=int, default=None, metavar="N",
                    help="cap read_csv at N rows -- a cheap sweep. Recorded per "
                         "entry; entries with a different sample size are re-measured")
    ap.add_argument("--timeout", type=float, default=3600,
                    help="seconds per pipeline before it is killed (default 3600)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used to run the pipelines")
    ap.add_argument("--force", action="store_true",
                    help="re-measure even pipelines already in the store")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-measure entries whose last run failed or timed out")
    ap.add_argument("--no-stats", action="store_true",
                    help="skip stratum's per-operator timing (wall clock only, "
                         "without the instrumentation overhead)")
    ap.add_argument("--mem-mode", default="process",
                    choices=("process", "system", "off"),
                    help="RSS sampling: 'process' this pipeline's interpreter, "
                         "'system' total memory in use (for workloads that fan out "
                         "over processes), 'off' to skip (default: process)")
    ap.add_argument("--mem-interval", type=float, default=0.1, metavar="SEC",
                    help="RSS sampling interval (default 0.1)")
    ap.add_argument("--no-mem-csv", action="store_true",
                    help="keep only the downsampled curve in the store, without "
                         "writing the full sample series per pipeline")
    ap.add_argument("--list", action="store_true",
                    help="show what the store holds and what a sweep would run")
    args = ap.parse_args(argv)

    pipe_dirs = [d.resolve() for d in (args.pipelines or [Path("pipelines")])]
    for d in pipe_dirs:
        if not d.is_dir():
            ap.error(f"no such directory: {d}")

    # Absolute: the child process runs with cwd=--run-in, so a relative store
    # path would make its --mem-csv resolve against a different directory than
    # the one this process created.
    store_path = (args.out or default_store_path(pipe_dirs[0])).resolve()
    store = load_store(store_path)

    pipelines = discover(pipe_dirs)
    if args.only:
        wanted = set(args.only)
        pipelines = [(n, p) for n, p in pipelines if n in wanted]
        unknown = wanted - {n for n, _ in pipelines}
        if unknown:
            ap.error(f"not found in {', '.join(map(str, pipe_dirs))}: "
                     f"{' '.join(sorted(unknown))}")
    if not pipelines:
        ap.error(f"no pipeline files in {', '.join(map(str, pipe_dirs))}")

    # Before deciding what is cached: which stratum built these numbers. The
    # stamp used to be written only at the end of a sweep and never read back,
    # so a store measured under an older stratum was silently reused as if it
    # were current.
    versions = _versions(args.python) or store["meta"].get("versions", {})
    stratum_commit = versions.get("stratum_commit")
    stored_commit = (store["meta"].get("versions") or {}).get("stratum_commit")
    if store["pipelines"] and stratum_commit and stored_commit != stratum_commit:
        print(f"! store was measured under stratum "
              f"{stored_commit[:12] if stored_commit else '(unrecorded)'}, this "
              f"interpreter has {stratum_commit[:12]} -- those entries are stale "
              f"and will be re-measured", file=sys.stderr)

    todo = []
    for name, path in pipelines:
        entry = store["pipelines"].get(name)
        if entry and not args.force:
            if is_fresh(entry, path, args.sample_rows, args.mem_mode,
                        stratum_commit):
                continue
            if entry.get("status") != "ok" and not args.retry_failed:
                continue
        todo.append((name, path))
    # ``--limit`` only shortens *this* run: the pipelines it drops are pending,
    # not cached, and counting them as cached made a batched sweep read as if it
    # were nearly finished after its first batch.
    pending = len(todo)
    if args.limit:
        todo = todo[:args.limit]
    cached = len(pipelines) - pending
    deferred = pending - len(todo)

    if args.list:
        print(f"store: {store_path}  ({len(store['pipelines'])} entry/entries)")
        where = f"{args.sample_rows:,} rows" if args.sample_rows else "full data"
        print(f"a sweep at {where} would run {pending} of {len(pipelines)}"
              f"{f' ({len(todo)} of them now, --limit {args.limit})' if deferred else ''}:")
        for name, path in pipelines:
            entry = store["pipelines"].get(name)
            mark = ("would run" if (name, path) in todo
                    else "pending  " if entry is None or
                         not is_fresh(entry, path, args.sample_rows,
                                      args.mem_mode, stratum_commit)
                    else "cached   ")
            print(f"  {mark}  {name:<24} {_fmt(entry) if entry else '—'}")
        return 0

    run_in = (args.run_in or _guess_run_in(pipe_dirs[0])).resolve()
    if not run_in.is_dir():
        ap.error(f"no such directory: {run_in}")
    _warn_about_data(run_in)

    print(f"{len(pipelines)} pipeline(s), {len(todo)} to measure "
          f"({cached} cached"
          f"{f', {deferred} left for a later batch' if deferred else ''})",
          file=sys.stderr)
    print(f"  run-in: {run_in}", file=sys.stderr)
    print(f"  store:  {store_path}", file=sys.stderr)
    if args.sample_rows:
        print(f"  sample: {args.sample_rows} rows", file=sys.stderr)

    store["version"] = STORE_VERSION
    mem_dir = None if args.no_mem_csv else store_path.with_suffix(".mem")
    store["meta"].update(
        pipelines=[str(d) for d in pipe_dirs], run_in=str(run_in),
        sample_rows=args.sample_rows, stats_enabled=not args.no_stats,
        cv_from_plan="stratum grid_search resolves mark_as_X(cv=...)",
        mem_mode=args.mem_mode, mem_interval_s=args.mem_interval,
        mem_csv_dir=str(mem_dir.name) if mem_dir else None,
        versions=versions,
    )

    failures = 0
    for i, (name, path) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {name} … ", end="", flush=True, file=sys.stderr)
        entry = run_one(path, run_in=run_in, python=args.python,
                        timeout=args.timeout, sample_rows=args.sample_rows,
                        stats=not args.no_stats, mem_mode=args.mem_mode,
                        mem_interval=args.mem_interval,
                        mem_csv=(mem_dir / f"{name}.csv") if mem_dir else None)
        entry["path"] = str(path.relative_to(pipe_dirs[0].parent)
                            if path.is_relative_to(pipe_dirs[0].parent) else path)
        # Per entry, not just per store: a sweep interrupted halfway leaves a
        # store whose entries came from two builds, and each has to be judged on
        # the one that produced it.
        entry["stratum_commit"] = stratum_commit

        store["pipelines"][name] = entry
        save_store(store_path, store)      # after each one: a kill keeps progress
        print(_fmt(entry), file=sys.stderr)
        failures += entry.get("status") != "ok"

    ok = sum(1 for e in store["pipelines"].values() if e.get("status") == "ok")
    print(f"\n{len(todo)} measured ({failures} failed) · store now holds "
          f"{ok}/{len(store['pipelines'])} ok · {store_path}", file=sys.stderr)
    return 1 if failures else 0


def _warn_about_data(run_in: Path) -> None:
    """Flag an unusable ``input/`` before spending a sweep discovering it.

    A dangling symlink is the failure that actually happens (a repo keeps the
    layout but not the data), and it looks identical to a present dataset until
    something opens the file.
    """
    data = run_in / "input"
    if not data.is_dir():
        print(f"! {run_in} has no input/ -- pipelines reading ./input will fail",
              file=sys.stderr)
        return
    entries = list(data.iterdir())
    broken = [e.name for e in entries if not e.exists()]
    if not entries:
        print(f"! {data} is empty", file=sys.stderr)
    elif broken:
        print(f"! {data}: dangling symlink(s): {' '.join(broken)}", file=sys.stderr)


def _guess_run_in(pipe_dir: Path) -> Path:
    """Nearest ancestor holding an ``input/`` folder; the pipelines' parent else."""
    for cand in (pipe_dir, *pipe_dir.parents):
        if (cand / "input").is_dir():
            return cand
    return pipe_dir.parent


if __name__ == "__main__":
    raise SystemExit(main())
