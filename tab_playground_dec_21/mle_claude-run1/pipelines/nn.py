"""Torch MLP wrapped as an sklearn estimator (via skorch) for skrub DataOps.

skrub's .skb.apply needs an sklearn-style estimator; skorch.NeuralNetClassifier
provides that around a torch nn.Module. TorchMLP adds the glue this task needs:
  - StandardScaler  : NNs need scaled inputs (trees didn't).
  - float32 cast    : StandardScaler emits float64; torch wants float32.
  - LabelEncoder    : the net trains on 0..C-1 (CrossEntropyLoss) but predicts in
                      the RAW 1..7 Cover_Type domain, so the harness scores it
                      exactly like LightGBM -- avoids the non-contiguous-label
                      problem that sank XGBoost (README).
  - train_split=None: no skorch-internal validation split, which would do a
                      stratified split and choke on the 1-row class 5.
"""
import os
# LightGBM (libomp) and torch (its own OpenMP) both load an OpenMP runtime; on macOS
# the second load aborts the process with no Python traceback. Allow the duplicate.
# Set BEFORE importing torch so it is in place when torch's libomp initialises.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline
from skorch import NeuralNetClassifier

from common import SEED


_CLIP = 30.0  # |z-score| beyond this is pathological; clip for NN input stability


def _sanitize_raw(a):
    """Raw guard: to float64, drop any non-finite (keeps StandardScaler happy)."""
    return np.nan_to_num(np.asarray(a, dtype="float64"), posinf=0.0, neginf=0.0)


def _post_scale(a):
    """After StandardScaler: clip extreme z-scores, drop non-finite, cast float32."""
    a = np.nan_to_num(np.asarray(a, dtype="float64"), posinf=0.0, neginf=0.0)
    np.clip(a, -_CLIP, _CLIP, out=a)
    return a.astype("float32")


class MLPModule(nn.Module):
    def __init__(self, n_features, n_classes=7, hidden=256, dropout=0.2, n_layers=2):
        super().__init__()
        blocks, d = [], n_features
        for _ in range(n_layers):
            blocks += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        blocks += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*blocks)

    def forward(self, X):
        return self.net(X)  # logits; CrossEntropyLoss / predict softmax handle the rest


def resolve_device(pref=None):
    if pref:
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TorchMLP(BaseEstimator, ClassifierMixin):
    """sklearn-compatible torch MLP: scale -> cast -> skorch net, raw-label domain."""

    def __init__(self, hidden=256, dropout=0.2, n_layers=2, lr=1e-3,
                 max_epochs=15, batch_size=8192, device=None):
        self.hidden = hidden
        self.dropout = dropout
        self.n_layers = n_layers
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y):
        torch.manual_seed(SEED)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        yenc = self.le_.transform(y).astype("int64")
        n_features = np.asarray(X).shape[1]
        net = NeuralNetClassifier(
            MLPModule,
            module__n_features=n_features,
            module__n_classes=len(self.classes_),
            module__hidden=self.hidden,
            module__dropout=self.dropout,
            module__n_layers=self.n_layers,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            iterator_train__drop_last=True,   # avoid a size-1 final batch breaking BatchNorm
            train_split=None,
            device=resolve_device(self.device),
            verbose=0,
        )
        self.pipe_ = make_pipeline(
            FunctionTransformer(_sanitize_raw),
            StandardScaler(),
            FunctionTransformer(_post_scale),
            net,
        )
        self.pipe_.fit(X, yenc)
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.pipe_.predict(X))

    def predict_proba(self, X):
        return self.pipe_.predict_proba(X)
