"""GPU listwise ranker: learned tracker embeddings + a softmax over all 355.

The first pipeline to beat the GBDT plateau, and it does so by attacking the
diagnosed bottleneck directly rather than adding another feature.

Two changes from every earlier pipeline, both aimed at rare trackers (recovered
38.5% of the time against 94.6% for common ones, carrying 76.5% of all misses):

  EMBEDDINGS SHARE STRENGTH. ~28k positives spread over 355 trackers leaves a
  rare tracker with almost no examples of its own, and a tree can only learn it
  from those. Here each candidate's logit is built from embeddings of its
  identity AND of its company, brand, category and country -- so a rare Google
  tracker inherits from every domain that uses any Google tracker. pipeline_07's
  `meta` block was the hand-built version of this idea and bought +0.002; making
  it a learned representation is the same idea with the weights fitted.

  A LISTWISE LOSS. Each domain is exactly one list of 355 candidates, so the
  loss is a softmax over the whole list with soft targets y/n_true: place the
  domain's probability mass on its true trackers, in competition with the other
  354. pipeline_06 showed lambdarank ties with log-loss; a full-list softmax is
  a stronger statement of the same idea, and one that only became cheap on a GPU.

ablation_03 swept the schedule; 40 epochs won and longer was clearly worse
(e40 0.87882, e80 0.87702, e120 0.87242), as was a wider embedding (e80_emb64
0.87458). The model's value is in the embeddings, not in capacity -- a 512x3 MLP
scored below the 256x2 default in the fold-0 probe too.

Runs on one A100 in ~20s per fold, i.e. FASTER than the LightGBM it beats
(~60s), because the whole training fold is only ~0.7 GB of float32 and lives
resident on the GPU.
"""
import skrub

from common import TorchListRanker, attach_scoring, features, load_xy

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior", "tld", "nbr_out", "nbr_in", "nbr_w",
                                     "meta", "host", "content", "trkcode"))
    feats = feats.skb.concat([X[["domain_id"]]], axis=1)
    pred = feats.skb.apply(TorchListRanker(epochs=40), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "GPU listwise ranker: tracker/company embeddings + 355-way softmax"
PARENT = "pipeline_09"
