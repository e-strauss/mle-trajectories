"""pipeline_23 -- MODEL: standalone torch MLP (skorch) on the champion features.

First read on a different model family. Covertype is tree-friendly, so the MLP is
unlikely to beat LightGBM alone -- the real goal is a decorrelated blend member
(tree + NN) as a route to 0.96. This measures where a modest MLP lands standalone,
on the same champion feature set (base14 + ELU) and same CV/scorer, so it is
directly comparable and ready to blend.

Modest first config: 2 x 256 hidden, dropout 0.2, Adam lr 1e-3, 15 epochs, batch
8192, on MPS (Mac GPU). Scaling + label remapping handled inside TorchMLP.
"""
from common import load_xy
from features import best_features
from nn import TorchMLP

X, y = load_xy(target="Cover_Type")
X = best_features(X.skb.drop(cols="Id"))
pred = X.skb.apply(
    TorchMLP(hidden=256, n_layers=2, dropout=0.2, lr=1e-3,
             max_epochs=15, batch_size=8192, device="mps"),
    y=y,
)

DESCRIPTION = "MODEL: torch MLP (skorch, 2x256, 15ep, mps) on champion features"
PARENT = "pipeline_21"
