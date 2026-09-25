"""Exploration round 6 -- price the four new blocks on one fold before spending pipelines.

pipeline_05 (rank features) and pipeline_06 (lambdarank) both came back flat, so
the model is not the bottleneck and the remaining gain has to come from features.
Exploration round 5 bounded the three unused graph signals by how much of the
miss set they can even reach: co-occurrence 13.9%, 2-hop 10.7%, co-citation 6.1%.
Reachability is only an upper bound though, so this measures each block's ACTUAL
marginal recall@10 on fold 0, on top of the pipeline_04 feature set.

One fold, so the numbers here are noisier than a scored pipeline (pipeline_04's
fold spread is ~0.002); this is for ordering the candidates, not for the board.
"""
import time

import numpy as np
import pandas as pd

from common import BLOCKS, INPUT, build_rows, load_context, make_cv, make_model, recall_at_10

BASE = ("prior", "tld", "nbr_out", "nbr_in", "nbr_w")
NEW = ["host", "content", "cooc", "hop2"]

print("loading ...")
ctx = load_context(INPUT)
rows = build_rows(ctx)
X = rows.drop(columns=["label"])
y = rows["label"]
cv = make_cv()
tr, te = next(iter(cv.split(X, y, groups=X["domain_id"])))

cache = {}


def block(name):
    if name not in cache:
        t0 = time.time()
        cache[name] = BLOCKS[name](ctx, X)
        print(f"  built {name:8s} {cache[name].shape[1]:2d} cols in {time.time() - t0:5.1f}s")
    return cache[name]


class _Fitted:
    def __init__(self, m, f):
        self.m, self.f = m, f

    def predict_proba(self, Xt):
        return self.m.predict_proba(self.f.loc[Xt.index])


def run(names):
    f = pd.concat([X[["tracker_id"]]] + [block(n) for n in names], axis=1)
    m = make_model().fit(f.iloc[tr], y.iloc[tr])
    return recall_at_10(_Fitted(m, f), X.iloc[te], y.iloc[te]), f.shape[1]


base_score, ncol = run(list(BASE))
print(f"\nbase (pipeline_04 features, {ncol} cols): fold-0 recall@10 = {base_score:.4f}")

print("\nmarginal value of each new block, added alone:")
scores = {}
for n in NEW:
    s, c = run(list(BASE) + [n])
    scores[n] = s - base_score
    print(f"  +{n:8s} ({c - ncol:2d} cols)  {s:.4f}   delta {s - base_score:+.4f}")

print("\nall four together:")
s, c = run(list(BASE) + NEW)
print(f"  +all       ({c - ncol:2d} cols)  {s:.4f}   delta {s - base_score:+.4f}")

order = sorted(scores, key=scores.get, reverse=True)
print("\ngreedy forward selection over the new blocks:")
cur, sel = base_score, list(BASE)
for n in order:
    s, _ = run(sel + [n])
    keep = s > cur
    print(f"  {'KEEP' if keep else 'drop'} {n:8s} {s:.4f}  (from {cur:.4f})")
    if keep:
        sel, cur = sel + [n], s
print("  selected:", [n for n in sel if n not in BASE], f"-> {cur:.4f}")
