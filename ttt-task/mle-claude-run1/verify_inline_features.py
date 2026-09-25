"""Equivalence check: inlined features == the original run's frozen parquet store.

The normalisation for this collection moved the feature computation out of
precomputed parquet files and into the pipeline plan (`pipelines/features.py`).
That is only legitimate if the numbers did not move: `results.json` records
scores that were produced against the ORIGINAL store, and they stay valid only
if the inlined builders reproduce it.

This is the analogue of the design-matrix check the other skrubified trajectory
in this collection runs. It rebuilds every block from `input/` and compares
against the store, column-for-column, for both the train and the target half.

    python verify_inline_features.py [--store /path/to/workspace/features]

Exit code 0 means every block matched.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "pipelines"))
import features  # noqa: E402

DEFAULT_STORE = ("/home/estrauss-ldap/repos/mle-claude/workspaces/"
                 "trackthetrackers-task_20260921_173740_290f/features")
N_TRK = 355
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def compare(name, built, store_path, ctx, split):
    """Compare one block's inlined frame against its stored parquet."""
    ref = pd.read_parquet(store_path)
    got = features.split_rows(built, ctx, split)
    if list(got.columns) != list(ref.columns):
        return f"column mismatch: {list(got.columns)[:4]}... vs {list(ref.columns)[:4]}..."
    if len(got) != len(ref):
        return f"row count {len(got)} vs {len(ref)}"
    if not np.array_equal(got["domain_id"].to_numpy(), ref["domain_id"].to_numpy()):
        return "domain_id order differs"
    for c in ref.columns:
        a, b = got[c].to_numpy(), ref[c].to_numpy()
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            a = a.astype(np.float64)
            b = b.astype(np.float64)
            bad = ~(np.isclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True))
            if bad.any():
                i = int(np.argmax(bad))
                return (f"column {c}: {int(bad.sum())} differing values, "
                        f"first at row {i}: {a[i]!r} vs {b[i]!r}")
        else:
            if not pd.Series(a).equals(pd.Series(b)):
                neq = pd.Series(a).ne(pd.Series(b)) & ~(pd.isna(a) & pd.isna(b))
                if neq.any():
                    i = int(np.argmax(neq.to_numpy()))
                    return f"column {c}: differs, first at row {i}: {a[i]!r} vs {b[i]!r}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--input", default=str(Path(__file__).resolve().parent / "input"))
    args = ap.parse_args()
    store = Path(args.store)
    if not store.is_dir():
        sys.exit(f"store not found: {store}\n"
                 "Pass --store <original workspace>/features. Without it there is "
                 "nothing to verify against; the inlined build still runs.")

    ctx = features.load_context(args.input)
    log(f"context: {ctx.n} rows ({ctx.n_train} train + {ctx.n - ctx.n_train} target)")

    # shared intermediates, built once, exactly as common.build_blocks wires them
    edges = features.seed_edges(ctx)
    log("seed edges built")
    linkdeg = features.link_degrees(ctx)
    log("link degrees built")
    labels = features.block_labels(ctx)
    hub = features.block_nbr_hub(ctx, edges, linkdeg)
    two_h = features.block_nbr_2h(ctx, edges, linkdeg)
    log("labels / nbr_hub / nbr_2h built")

    built = {
        "labels": labels,
        "nbr_out": features.block_nbr_out(ctx, edges),
        "nbr_in": features.block_nbr_in(ctx, edges),
        "tld_pop": features.block_tld_pop(ctx, labels),
        "nbr_hub": hub,
        "nbr_rec": features.block_nbr_rec(ctx, edges),
        "nbr_2h": two_h,
        "nbr_frac": features.block_nbr_frac(ctx, edges),
        "direct": features.block_direct(ctx),
        "trk_cooc": features.block_trk_cooc(ctx, hub),
        "tok_pop": features.block_tok_pop(ctx),
        "meta": features.block_meta(ctx, edges),
        "meta2": features.block_meta2(ctx, edges, linkdeg, hub, two_h),
    }
    log("all blocks built")

    failures = []
    print(f"\n{'block':12s} {'split':7s} result")
    print("-" * 62)
    for name, frame in built.items():
        splits = ["train"] if name == "labels" else ["train", "target"]
        for split in splits:
            path = store / f"{name}_{split}.parquet"
            if not path.is_file():
                print(f"{name:12s} {split:7s} SKIP (not in store)")
                continue
            err = compare(name, frame, path, ctx, split)
            print(f"{name:12s} {split:7s} {'OK' if err is None else 'MISMATCH -- ' + err}")
            if err is not None:
                failures.append((name, split, err))

    print("-" * 62)
    if failures:
        print(f"{len(failures)} MISMATCH(es) -- the inlined build is NOT equivalent; "
              "results.json cannot be carried over as-is.")
        return 1
    print("all blocks identical to the original frozen store; the recorded "
          "scores in pipelines/results.json carry over unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
