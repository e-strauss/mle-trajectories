"""Round 6: offline GPU-MLP sweep on fold 0 -- pick an architecture and objective.

Exploration only (nothing recorded): trains `models.TorchMLPRanker` on fold 0 of
the workspace CV so a scored pipeline can start from a sane configuration
instead of burning fused-choice runs on obviously-bad ones. Also the first test
of the metric-aligned `softmax_ce` objective (E[y_t/n_true|x]) against plain
`bce` (P(y_t=1|x)) -- see the TorchMLPRanker docstring.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import recall_at_k
from models import TorchMLPRanker

WS = Path(__file__).resolve().parent.parent
FEAT = WS / "features"
N_TRK = 355
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


lab = pd.read_parquet(FEAT / "labels_train.parquet")
Y = lab[[f"y_{i}" for i in range(N_TRK)]].to_numpy(dtype=np.float32)
parts = []
for name, pref in (("nbr_out", "no_"), ("nbr_in", "ni_"), ("tld_pop", "tp_")):
    M = pd.read_parquet(FEAT / f"{name}_train.parquet")[
        [f"{pref}{i}" for i in range(N_TRK)]].to_numpy(dtype=np.float32)
    if pref != "tp_":
        M = M / np.maximum(M.sum(axis=1, keepdims=True), 1.0)
    parts.append(M)
meta = pd.read_parquet(FEAT / "meta_train.parquet")
deg = np.log1p(meta[["outdeg", "indeg", "n_trk_nbr_out", "n_trk_nbr_in",
                     "n_trk_nbr_tot"]].to_numpy(dtype=np.float32))
num = meta[["n_labels", "host_len", "n_digits", "n_hyphens"]].to_numpy(np.float32)
fp = meta["freedom_of_the_press"].fillna(meta["freedom_of_the_press"].median()) \
    .to_numpy(dtype=np.float32)[:, None]
X = np.hstack(parts + [deg, num, fp])
del parts
log("X", X.shape)

tr, te = next(iter(KFold(3, shuffle=True, random_state=42).split(X)))
sc = StandardScaler().fit(X[tr])
Xtr, Xte = sc.transform(X[tr]).astype(np.float32), sc.transform(X[te]).astype(np.float32)
log("fold 0 split", Xtr.shape, Xte.shape)

CONFIGS = {
    "bce_1024x512_e12":      dict(hidden=(1024, 512), epochs=12, loss="bce"),
    "sce_1024x512_e12":      dict(hidden=(1024, 512), epochs=12, loss="softmax_ce"),
    "sce_1024x512_e30":      dict(hidden=(1024, 512), epochs=30, loss="softmax_ce"),
    "sce_2048x1024_e30_d3":  dict(hidden=(2048, 1024), epochs=30, dropout=0.3,
                                  loss="softmax_ce"),
    "sce_512_e30":           dict(hidden=(512,), epochs=30, loss="softmax_ce"),
    "bce_1024x512_e30":      dict(hidden=(1024, 512), epochs=30, loss="bce"),
    "bce_1024x512_e30_pw10": dict(hidden=(1024, 512), epochs=30, loss="bce",
                                  pos_weight=10.0),
    "sce_2048x1024x512_e40": dict(hidden=(2048, 1024, 512), epochs=40, dropout=0.3,
                                  loss="softmax_ce"),
}
for name, cfg in CONFIGS.items():
    t = time.time()
    m = TorchMLPRanker(seed=0, **cfg).fit(Xtr, Y[tr])
    r = recall_at_k(Y[te], m.predict(Xte))
    log(f"{name:24s} recall@10={r:.5f}  ({time.time() - t:.0f}s)")
