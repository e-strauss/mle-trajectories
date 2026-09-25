"""Seed-averaged neural ranker across all four A100s, blended with LightGBM.

A confirmation run for a negative result, scored under the real 3-fold CV
because every delta involved is at the ~0.001 noise floor and one fold cannot
settle it.

The premise was that a listwise softmax fitted on only 9.5k training lists, with
an embedding table whose gradients come from a long-tailed label distribution,
would be a high-variance estimator worth averaging. That premise is false, and
the measurement that killed it is the within-domain rank correlation between
independently fitted members:

    seed B vs seed A                  0.9943
    seed C vs seed A                  0.9943
    60 epochs, dropout 0.25           0.9924
    25 epochs                         0.9850
    emb 16 / hidden 128               0.9794
    emb 64 / hidden 384 / 3 layers    0.9785
    LightGBM                          0.7539   <- for scale

Changing the seed, the width, the depth and the schedule all leave the model
ranking candidates the same way. There is no variance to average: on fold 0,
1 / 2 / 4 / 8 / 16 seeds scored 0.8789 / 0.8784 / 0.8780 / 0.8790 / 0.8781, and
ensembling three architectures or all seven variants gave 0.8787 / 0.8790 --
one net, every time.

That correlation table is also the real explanation for why pipeline_12's blend
worked: not "ensembling helps" but "these two model families genuinely disagree,
at 0.754". Averaging things that agree at 0.99 buys nothing no matter how many
GPUs it runs on.

Kept as a scored pipeline rather than a note because a confirmed negative under
the full CV is worth as much as a win here -- it closes off the whole
"more of the same model" direction.

Uses 8 seeds across 4 devices, ~29s per fold wall-clock against ~20s for a
single fit, so the parallelism itself works; it is the averaging that does not.
"""
import skrub

from common import HybridRanker, attach_scoring, features, load_xy

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior", "tld", "nbr_out", "nbr_in", "nbr_w",
                                     "meta", "host", "content", "trkcode"))
    feats = feats.skb.concat([X[["domain_id"]]], axis=1)
    pred = feats.skb.apply(HybridRanker(nn_weight=2 / 3, n_seeds=8), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "8-seed GPU ranker (4x A100) + LightGBM blend -- tests seed averaging"
PARENT = "pipeline_12"
