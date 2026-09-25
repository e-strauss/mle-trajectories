"""Estimators for the cold-start multi-label tracker-ranking task.

All of them are *rankers*: `predict(X)` returns an (n, 355) score matrix whose
top-10 per row is the submission, which is what `common.recall_at_k` scores.
They subclass `RegressorMixin` FIRST so sklearn's tag system reports them as
predictors (guide pitfall 16) and skrub treats `.skb.apply(est, y=y)` as a
supervised prediction node rather than a transformer.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

N_TRK = 355


def _y2mat(y):
    """(n, 355) float32 label matrix from a DataFrame / array target."""
    return np.asarray(y, dtype=np.float32)


def _block(X, prefix):
    """The (n, 355) sub-matrix of X whose columns start with `prefix`.

    Returns None when the block is absent, so one estimator can serve pipelines
    that joined different feature blocks.
    """
    cols = [c for c in X.columns if c.startswith(prefix)]
    if not cols:
        return None
    cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
    return X[cols].to_numpy(dtype=np.float32)


class PopularityRanker(RegressorMixin, BaseEstimator):
    """Constant baseline: rank every domain by global tracker frequency.

    Learned from the training fold's labels only (no leakage), which is exactly
    "always guess the 10 most common trackers".
    """

    def fit(self, X, y):
        self.prior_ = _y2mat(y).mean(axis=0)
        return self

    def predict(self, X):
        return np.tile(self.prior_, (len(X), 1))


class BlendRanker(RegressorMixin, BaseEstimator):
    """Hand-weighted blend of the precomputed score blocks -- no learning.

    score = w_out*f(no) + w_in*f(ni) + w_tld*tp + w_pop*prior
    where f is log1p (counts are heavy-tailed) or l1-normalisation, and `prior`
    is the global tracker frequency of the training fold. Missing blocks are
    skipped, so the same class expresses the popularity-only and
    neighbourhood-only variants.
    """

    def __init__(self, w_out=1.0, w_in=1.0, w_tld=1.0, w_pop=1.0, transform="log1p"):
        self.w_out = w_out
        self.w_in = w_in
        self.w_tld = w_tld
        self.w_pop = w_pop
        self.transform = transform

    def _f(self, M):
        if self.transform == "log1p":
            return np.log1p(M)
        if self.transform == "l1":
            return M / np.maximum(M.sum(axis=1, keepdims=True), 1.0)
        if self.transform == "sqrt":
            return np.sqrt(M)
        if self.transform == "raw":
            return M
        raise ValueError(self.transform)

    def fit(self, X, y):
        self.prior_ = _y2mat(y).mean(axis=0)
        return self

    def predict(self, X):
        out = np.tile(self.w_pop * self.prior_, (len(X), 1))
        for prefix, w in (("no_", self.w_out), ("ni_", self.w_in)):
            M = _block(X, prefix)
            if M is not None and w:
                out += w * self._f(M)
        M = _block(X, "tp_")
        if M is not None and self.w_tld:
            out += self.w_tld * M
        return out


class BlockTransform(TransformerMixin, BaseEstimator):
    """Squash the heavy-tailed 355-wide count blocks; pass every other column through.

    The neighbour blocks are raw counts whose row sums span 0 to >10^5 (degree
    varies wildly), so a linear model or MLP needs them on a comparable scale:
      l1     -> each block becomes a distribution over the 355 trackers
                (what pipeline_02 found best), i.e. "what fraction of my
                neighbours' tracker mentions are tracker t"
      log1p  -> keeps degree information but compresses the tail
      raw    -> no change (ablation)
    `tp_*` is already a probability and is never touched.
    """

    def __init__(self, mode="l1", prefixes=("no_", "ni_")):
        self.mode = mode
        self.prefixes = prefixes

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        pos = {c: i for i, c in enumerate(self.feature_names_in_)}
        self.groups_ = {p: np.array([pos[c] for c in self.feature_names_in_
                                     if c.startswith(p)], dtype=np.int64)
                        for p in self.prefixes}
        return self

    def transform(self, X):
        # one contiguous float32 copy + numpy column slices: doing this via
        # DataFrame assignment on a 1000+ column frame dominated the CV wall
        # clock (it is re-run for every fold and grid variant).
        M = X.to_numpy(dtype=np.float32, copy=True)
        for idx in self.groups_.values():
            if idx.size == 0:
                continue
            sub = M[:, idx]
            if self.mode == "l1":
                sub = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1.0)
            elif self.mode == "log1p":
                sub = np.log1p(sub)
            elif self.mode != "raw":
                raise ValueError(self.mode)
            M[:, idx] = sub
        return pd.DataFrame(M, columns=self.feature_names_in_,
                            index=getattr(X, "index", None))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


class TorchMLPRanker(RegressorMixin, BaseEstimator):
    """GPU multi-label MLP: 355 sigmoid heads trained with BCE, `predict` = logits.

    Hand-rolled rather than skorch (guide section 14) because the target here is
    a (n, 355) multi-label MATRIX and the natural loss is
    `BCEWithLogitsLoss` over all heads at once -- skorch's NeuralNetClassifier
    assumes a 1-D class target, and NeuralNetRegressor would need the same
    manual y handling anyway. The three things the guide warns about are handled
    by the plan (BlockTransform + StandardScaler give a clean, scaled float32
    matrix) plus the float32 cast here.

    `predict` returns raw logits: Recall@10 only needs the per-row ranking, and
    the sigmoid is monotone, so logits and probabilities rank identically.

    `loss` picks the training objective, and this matters for Recall@10:

      "bce"        355 independent sigmoids on the raw 0/1 labels -> estimates
                   P(y_t = 1 | x).
      "softmax_ce" softmax over the 355 trackers, cross-entropy against the
                   row-normalised target y_t / n_true (which sums to 1) ->
                   estimates E[y_t / n_true | x].

    The second is the metric-aligned one. Recall@10 of a chosen set S is
    sum_{t in S} y_t / n_true, so its expectation is maximised by taking the 10
    largest E[y_t / n_true | x] -- NOT the 10 largest P(y_t = 1 | x). The two
    rankings differ because n_true is correlated with which trackers appear: a
    tracker that mostly shows up on 20-tracker portals contributes less recall
    per slot than one that is the sole tracker of a small site. Both objectives
    are fitted on the training fold's labels only.

    pos_weight upweights positives in "bce" (labels are sparse: ~2 of 355/row).

    `standardize=True` z-scores the inputs on the GPU from the TRAINING fold's
    own mean/std (refit per fold, so leakage-free -- the same computation a
    plan-side StandardScaler does). It lives here because an sklearn
    SimpleImputer + StandardScaler over the ~1300-column float64 frame
    dominated pipeline_05's wall clock and is re-run for every (variant, fold)
    pair of a fused-choice grid; in torch it is two reductions.

    `n_seeds > 1` trains that many independent networks (seeds seed..seed+n-1)
    and averages their per-row probability distributions. A single MLP's
    Recall@10 wobbles by a few 1e-4 with the init/shuffle seed, which is the
    same order as the differences between the better configurations, so
    averaging removes that wobble rather than gambling on one draw.
    """

    def __init__(self, hidden=(1024, 512), dropout=0.2, lr=1e-3, epochs=12,
                 batch_size=4096, weight_decay=1e-5, pos_weight=1.0,
                 loss="bce", standardize=False, n_seeds=1, device="cuda",
                 seed=0, verbose=False):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.loss = loss
        self.standardize = standardize
        self.n_seeds = n_seeds
        self.device = device
        self.seed = seed
        self.verbose = verbose

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _mat(X):
        return (X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy")
                else np.asarray(X, dtype=np.float32))

    def _build(self, n_in, n_out):
        import torch.nn as nn
        layers, prev = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers += [nn.Linear(prev, n_out)]
        return nn.Sequential(*layers)

    # -- sklearn API ------------------------------------------------------
    def fit(self, X, y):
        import torch
        import torch.nn as nn

        dev = torch.device(self.device if torch.cuda.is_available() else "cpu")
        Xn, Yn = self._mat(X), _y2mat(y)
        self.n_features_in_ = Xn.shape[1]
        self.device_ = dev
        self.nets_ = []

        # whole sample lives on the GPU: 337k x ~1100 float32 is ~1.5 GB
        Xt = torch.from_numpy(np.ascontiguousarray(Xn)).to(dev)
        Yt = torch.from_numpy(Yn).to(dev)
        if self.standardize:
            self.mu_ = Xt.mean(dim=0, keepdim=True)
            self.sd_ = Xt.std(dim=0, keepdim=True).clamp_min(1e-6)
            Xt = (Xt - self.mu_) / self.sd_
        if self.loss == "bce":
            pw = torch.full((Yn.shape[1],), float(self.pos_weight), device=dev)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        elif self.loss == "softmax_ce":
            Yt = Yt / Yt.sum(dim=1, keepdim=True).clamp_min(1.0)   # rows sum to 1
            log_softmax = nn.LogSoftmax(dim=1)

            def loss_fn(logits, target):
                return -(target * log_softmax(logits)).sum(dim=1).mean()
        else:
            raise ValueError(self.loss)

        steps = self.epochs * max(1, (len(Xt) + self.batch_size - 1) // self.batch_size)
        for si in range(self.n_seeds):
            sd = self.seed + si
            torch.manual_seed(sd)
            np.random.seed(sd)
            net = self._build(Xn.shape[1], Yn.shape[1]).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr,
                                                        total_steps=steps)
            g = torch.Generator(device="cpu").manual_seed(sd)
            net.train()
            for ep in range(self.epochs):
                perm = torch.randperm(len(Xt), generator=g).to(dev)
                tot = 0.0
                for i in range(0, len(perm), self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    opt.zero_grad(set_to_none=True)
                    loss = loss_fn(net(Xt[idx]), Yt[idx])
                    loss.backward()
                    opt.step()
                    sched.step()
                    tot += float(loss.detach()) * len(idx)
                if self.verbose:
                    print(f"  seed {sd} epoch {ep + 1}/{self.epochs} "
                          f"loss={tot / len(Xt):.5f}", flush=True)
            net.eval()
            self.nets_.append(net)
        del Xt, Yt
        torch.cuda.empty_cache()
        return self

    def predict(self, X):
        """(n, 355) ranking scores. One net -> raw logits; several -> the mean
        predicted distribution (probabilities, not logits, is the right average
        across independently-initialised nets)."""
        import torch
        Xn = self._mat(X)
        out = np.empty((len(Xn), 355), dtype=np.float32)
        bs = 65536
        with torch.no_grad():
            for i in range(0, len(Xn), bs):
                xb = torch.from_numpy(
                    np.ascontiguousarray(Xn[i:i + bs])).to(self.device_)
                if self.standardize:
                    xb = (xb - self.mu_) / self.sd_
                if len(self.nets_) == 1:
                    z = self.nets_[0](xb)
                else:
                    acc = None
                    for net in self.nets_:
                        lg = net(xb)
                        p = (torch.softmax(lg, dim=1) if self.loss == "softmax_ce"
                             else torch.sigmoid(lg))
                        acc = p if acc is None else acc + p
                    z = acc / len(self.nets_)
                out[i:i + bs] = z.cpu().numpy()
        return out
