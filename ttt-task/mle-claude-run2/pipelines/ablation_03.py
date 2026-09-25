"""Ablation (fused choice): architecture / schedule for the GPU listwise ranker.

A single-fold probe already put TorchListRanker at 0.8789 against the tuned
GBDT's 0.8735 on the same fold, so the question is no longer whether the model
family helps but how much of it to buy. Training length mattered most in the
probe (20 epochs 0.8776 -> 40 epochs 0.8789) while a wider, deeper MLP was worse
(512x3: 0.8762), which is the usual shape for a model whose value is in its
embeddings rather than in its capacity -- so the grid pushes on epochs and
embedding width, not on MLP size.

Whole-estimator variants, one per configuration, for the same reason as
ablation_02: a readable list of real configurations beats a cross product.
"""
import skrub

from common import TorchListRanker, attach_scoring, features, load_xy

BLOCKS = ("prior", "tld", "nbr_out", "nbr_in", "nbr_w", "meta", "host",
          "content", "trkcode")

MODELS = {
    "e40":          dict(epochs=40),
    "e80":          dict(epochs=80),
    "e120":         dict(epochs=120),
    "e80_emb64":    dict(epochs=80, emb_dim=64),
    "e80_drop02":   dict(epochs=80, dropout=0.2),
}

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=BLOCKS)
    # the listwise loss groups rows by domain, so the key rides in the features
    feats = feats.skb.concat([X[["domain_id"]]], axis=1)
    preds = {name: feats.skb.apply(TorchListRanker(**kw), y=y)
             for name, kw in MODELS.items()}
    pred = skrub.choose_from(preds, name="arch").as_data_op()
    pred = attach_scoring(pred)

DESCRIPTION = "ABLATION: GPU listwise-ranker architecture / schedule sweep"
PARENT = "pipeline_09"
