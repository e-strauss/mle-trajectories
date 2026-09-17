"""Skrubified `pipelines/1/train0.py` (APTOS 2019 blindness detection).

The original trains two timm backbones on retina images, tracks a quadratic
weighted kappa (QWK) per epoch on a hold-out split, keeps the best epoch's
validation predictions of each model, averages them 50/50 and reports the QWK of
an `OptimizedRounder` fitted on that average.

Translation notes (every deviation is flagged at its site as well):

* The torch training loop is not a recorded operation, so it lives in a wrapper
  estimator -- but the loop itself is delegated to **skorch**, which gives the
  `nn.Module` a scikit-learn `fit`/`predict` API. The wrapper is the thin layer
  that turns a DataFrame of `id_code`s into a torch `Dataset` and decodes the
  network output; the epoch loop, the LR schedule and the per-epoch validation
  scoring are skorch callbacks.
* The original selects the best epoch on the very rows it then reports as its
  score, and fits the `OptimizedRounder` thresholds on those same rows. The
  second leak is part of the reported metric, so it is reproduced inside the
  scorer. The first one is not reproducible under cross-validation (the fold's
  validation rows are invisible at fit time), so the eval set is carved out of
  the fold's OWN training rows with the guide's `GetXY` pattern -- visible in
  the plan as its own node, and honest where the original was leaky.
* The original's split is a stock `StratifiedShuffleSplit(n_splits=1,
  test_size=0.2, random_state=42)`, which `mark_as_X` takes directly: the plan
  scores exactly the rows the original scores.
* Dropped, as they produce no score: nothing. The original writes no submission.
"""

import copy

import numpy as np
import pandas as pd
import skrub
import timm
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from scipy.optimize import minimize
from skorch import NeuralNetRegressor
from skorch.dataset import unpack_data
from skorch.callbacks import Callback, EpochScoring, LRScheduler
from skorch.helper import predefined_split
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# The original seeds once, at the top of main(); do the same once at import, so
# both models draw from one RNG stream exactly as they did there.
seed_everything(42)

# Configuration -- verbatim from the original.
INPUT_DIR = Path("./input")
if not INPUT_DIR.exists():
    INPUT_DIR = Path(".")

TRAIN_CSV = INPUT_DIR / "train.csv"
TRAIN_IMAGES_DIR = INPUT_DIR / "train_images"

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset -- the original's, keyed by arrays instead of a DataFrame so that the
# wrapper can build it from whatever rows a CV fold hands it.
# ---------------------------------------------------------------------------
class RetinopathyDataset(Dataset):
    def __init__(self, image_ids, labels, img_dir, transform=None):
        self.image_ids = np.asarray(image_ids)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.img_dir / f"{img_id}.png"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ---------------------------------------------------------------------------
# Loss functions and models -- verbatim from the original.
# ---------------------------------------------------------------------------
class OrdinalClassificationLoss(nn.Module):
    """Extended Binary Cross Entropy loss for ordinal targets: y > k for k in [0, K-2]"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, num_classes=5):
        levels = torch.arange(num_classes - 1, device=targets.device).unsqueeze(0)
        binary_targets = (targets.unsqueeze(1) > levels).float()
        return self.bce(logits, binary_targets)


class ConvNeXtV2RegressionModel(nn.Module):
    def __init__(self, model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", pretrained=True, drop_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)


class EVA02OrdinalModel(nn.Module):
    def __init__(self, model_name="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes - 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


def ordinal_logits_to_score(logits):
    """The original's `EVA02OrdinalModel.predict_continuous`, on numpy.

    `predict_continuous` is not reachable through skorch (which always calls
    `forward`), so the decoding moves to the estimator's `predict`. Same
    arithmetic: sum of the per-level sigmoids.
    """
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    return probs.sum(axis=1)


class OptimizedRounder:
    """Verbatim from the original -- thresholds fitted by Powell on QWK."""
    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5, 3.5]

    def _loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred < coef[1]:
                X_p[i] = 1
            elif pred < coef[2]:
                X_p[i] = 2
            elif pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return -cohen_kappa_score(y, X_p, weights="quadratic")

    def fit(self, X, y):
        res = minimize(self._loss, self.coef_, args=(X, y), method="Powell")
        self.coef_ = res.x

    def predict(self, X):
        X_p = np.copy(X)
        res = np.zeros_like(X_p, dtype=int)
        res[X_p >= self.coef_[0]] = 1
        res[X_p >= self.coef_[1]] = 2
        res[X_p >= self.coef_[2]] = 3
        res[X_p >= self.coef_[3]] = 4
        return res


def optimized_qwk(y_true, y_pred):
    """QWK after fitting the rounder's thresholds ON THE SCORED ROWS.

    This is the original's final metric, leak included: `final_rounder` is fitted
    on `ensemble_val_preds` / `val_targets` and then scores those same rows. It
    is part of what the original reports, so it is reproduced rather than fixed;
    the same function drives the per-epoch model selection below.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rounder = OptimizedRounder()
    rounder.fit(y_pred, y_true)
    return cohen_kappa_score(y_true, rounder.predict(y_pred), weights="quadratic")


QWK_SCORER = make_scorer(optimized_qwk, response_method="predict", greater_is_better=True)


# ---------------------------------------------------------------------------
# skorch plumbing
# ---------------------------------------------------------------------------
class AmpNeuralNetRegressor(NeuralNetRegressor):
    """skorch has no built-in AMP, and the original trains and infers under
    `torch.cuda.amp.autocast()` with a `GradScaler`. Both are kept here:
    `infer` supplies the autocast context (training forward *and* inference, as
    in the original), `train_step` the loss scaling. Everything is a no-op when
    CUDA is unavailable, exactly as the original's `GradScaler()` is.
    """

    def initialize(self):
        super().initialize()
        self.amp_enabled_ = torch.cuda.is_available()
        self.scaler_ = torch.amp.GradScaler("cuda", enabled=self.amp_enabled_)
        return self

    def infer(self, x, **fit_params):
        with torch.autocast("cuda", enabled=getattr(self, "amp_enabled_", False)):
            return super().infer(x, **fit_params)

    def train_step(self, batch, **fit_params):
        # skorch's own train_step, with scaler.scale/step/update replacing the
        # bare loss.backward() / optimizer.step() pair.
        self._zero_grad_optimizer()
        Xi, yi = unpack_data(batch)
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        self.scaler_.scale(loss).backward()
        self.notify(
            "on_grad_computed",
            named_parameters=list(self.get_all_learnable_params()),
            batch=batch,
            training=True,
        )
        for name in self._optimizers:
            self.scaler_.step(getattr(self, name + "_"))
        self.scaler_.update()
        return {"loss": loss, "y_pred": y_pred}


class RestoreBestEpoch(Callback):
    """Keep the weights of the epoch with the best monitored score.

    The original keeps `best_val_preds_i` from the epoch with the highest QWK;
    since predictions here are produced later, on the fold's real validation
    rows, the equivalent is to restore that epoch's weights at the end of
    training. Kept in memory rather than through `Checkpoint`, so that folds do
    not contend for a file on disk.
    """

    def __init__(self, monitor="valid_qwk"):
        self.monitor = monitor

    def on_train_begin(self, net, **kwargs):
        self.best_score_ = -np.inf
        self.best_state_ = None

    def on_epoch_end(self, net, **kwargs):
        score = net.history[-1, self.monitor]
        if score > self.best_score_:      # strict >, as in the original
            self.best_score_ = score
            self.best_state_ = copy.deepcopy(net.module_.state_dict())

    def on_train_end(self, net, **kwargs):
        if self.best_state_ is not None:
            net.module_.load_state_dict(self.best_state_)


class SkorchImageRegressor(RegressorMixin, BaseEstimator):
    """A timm backbone trained by skorch, fed by `id_code`s from the plan.

    This is the one thing no recorded operation can express (guide pitfall 19):
    a torch training loop. Everything the wrapper does beyond that -- building
    the `Dataset` from the fold's rows, decoding the network output -- is the
    glue between a DataFrame of image ids and torch, not hidden pipeline logic.
    The inner eval split is NOT made here: it arrives through `fit_kwargs` from
    the `EvalSetSplit` node, so it stays visible in the plan.
    """

    def __init__(self, module=None, module_kwargs=None, criterion=None,
                 predict_decoder=None, train_transform=None, val_transform=None,
                 epochs=4, lr=2e-4, weight_decay=1e-2, batch_size=BATCH_SIZE,
                 num_workers=4, images_dir=TRAIN_IMAGES_DIR, device=DEVICE):
        self.module = module
        self.module_kwargs = module_kwargs
        self.criterion = criterion
        self.predict_decoder = predict_decoder
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.images_dir = images_dir
        self.device = device

    # -- helpers -----------------------------------------------------------
    def _dataset(self, X, y, transform):
        labels = np.zeros(len(X), dtype=np.float32) if y is None else np.asarray(y, dtype=np.float32)
        return RetinopathyDataset(np.asarray(X["id_code"]), labels, self.images_dir, transform)

    def _decode(self, raw):
        if self.predict_decoder is not None:
            raw = self.predict_decoder(raw)
        return np.asarray(raw, dtype=np.float64).reshape(-1)

    def _epoch_qwk(self, net, X, y):
        """`EpochScoring` hook: the original's per-epoch OptimizedRounder + QWK."""
        return optimized_qwk(y, self._decode(net.predict(X)))

    # -- estimator API -----------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        train_ds = self._dataset(X, y, self.train_transform)

        callbacks = [
            # CosineAnnealingLR(T_max=epochs, eta_min=1e-6), stepped per epoch.
            ("lr_scheduler", LRScheduler(
                policy=torch.optim.lr_scheduler.CosineAnnealingLR,
                T_max=self.epochs, eta_min=1e-6, step_every="epoch")),
        ]
        if X_val is None:
            # No eval set (predict/score mode never reaches fit, but keep the
            # estimator usable stand-alone): train all epochs, keep the last.
            valid_split = None
        else:
            # The eval rows use the VALIDATION transform, as in the original --
            # which is why the split is a predefined one over a second Dataset
            # rather than skorch's ValidSplit over the training Dataset.
            valid_split = predefined_split(self._dataset(X_val, y_val, self.val_transform))
            callbacks += [
                # use_caching=True is required, not an optimisation: with a
                # plain torch Dataset (not a skorch one) EpochScoring can only
                # recover the eval targets from the cached batches -- without it
                # the scoring callable is handed y_test=None.
                ("valid_qwk", EpochScoring(self._epoch_qwk, name="valid_qwk",
                                           lower_is_better=False, on_train=False,
                                           use_caching=True)),
                ("restore_best", RestoreBestEpoch(monitor="valid_qwk")),
            ]

        self.net_ = AmpNeuralNetRegressor(
            module=self.module,
            **{f"module__{k}": v for k, v in (self.module_kwargs or {}).items()},
            criterion=self.criterion,
            optimizer=torch.optim.AdamW,
            optimizer__weight_decay=self.weight_decay,
            lr=self.lr,
            max_epochs=self.epochs,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            iterator_train__num_workers=self.num_workers,
            iterator_train__pin_memory=True,
            iterator_valid__shuffle=False,
            iterator_valid__num_workers=self.num_workers,
            iterator_valid__pin_memory=True,
            train_split=valid_split,
            callbacks=callbacks,
            device=self.device,
            verbose=0,
        )
        self.net_.fit(train_ds, y=None)
        return self

    def predict(self, X):
        return self._decode(self.net_.predict(self._dataset(X, None, self.val_transform)))


class EvalSetSplit(TransformerMixin, BaseEstimator):
    """Carve the per-epoch eval set out of THIS fold's training rows.

    The guide's section-7 `GetXY`: the original validated (and picked its best
    epoch) on the rows it reports as its score, which cross-validation cannot
    see at fit time. Splitting the fold's own training rows keeps the mechanism
    and drops the leak. Stratified on the target, like the original's outer
    StratifiedKFold.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
        return {"X": X_fit, "X_val": X_val, "y": y_fit, "y_val": y_val}

    def transform(self, X):
        # Predict mode: all rows, no eval set -- same KEYS, None for the
        # fit-only pieces (guide section 7).
        return {"X": X, "X_val": None, "y": None, "y_val": None}


with skrub.config_context(eager_data_ops=False):
    # 1. Load data -- recorded read of the original's TRAIN_CSV.
    data = skrub.as_data_op(str(TRAIN_CSV)).skb.apply_func(pd.read_csv)

    # 2. Mark the RAW target and the design matrix. X keeps `id_code`, which is
    #    what the Dataset resolves into an image path. The CV splitter is the
    #    original's own StratifiedShuffleSplit -- `train_test_split`-shaped
    #    splitters translate to an IDENTICAL partition, so the plan scores the
    #    same rows the original does. It lives here and nowhere else.
    y = data["diagnosis"].skb.mark_as_y()
    X = data.drop(columns=["diagnosis"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Carve the per-epoch eval set out of the fold's own training rows.
    X_y = X.skb.apply(EvalSetSplit(test_size=0.2, random_state=42), y=y, how="no_wrap")
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y)          # the default covers predict/score mode
    X_val, y_val = X_y["X_val"], X_y["y_val"]

    # 4. Model 1: ConvNeXtV2 regression (SmoothL1), img_size 384, 4 epochs.
    #    Transform lists are the original's, verbatim.
    train_transform_1 = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform_1 = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model_1 = SkorchImageRegressor(
        module=ConvNeXtV2RegressionModel,
        module_kwargs={"model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
                       "pretrained": True, "drop_rate": 0.2},
        criterion=nn.SmoothL1Loss,
        predict_decoder=None,                 # the module already returns a scalar
        train_transform=train_transform_1,
        val_transform=val_transform_1,
        epochs=4, lr=2e-4, weight_decay=1e-2,
    )
    pred_1 = X_fit.skb.apply(model_1, y=y_fit,
                             fit_kwargs={"X_val": X_val, "y_val": y_val})

    # 5. Model 2: EVA-02 ordinal classification, img_size 448, 4 epochs.
    train_transform_2 = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform_2 = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model_2 = SkorchImageRegressor(
        module=EVA02OrdinalModel,
        module_kwargs={"model_name": "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
                       "num_classes": 5, "pretrained": True},
        criterion=OrdinalClassificationLoss,
        predict_decoder=ordinal_logits_to_score,
        train_transform=train_transform_2,
        val_transform=val_transform_2,
        epochs=4, lr=5e-5, weight_decay=0.05,
    )
    pred_2 = X_fit.skb.apply(model_2, y=y_fit,
                             fit_kwargs={"X_val": X_val, "y_val": y_val})

    # 6. Ensemble: the original's 0.5 / 0.5 average of the two models'
    #    continuous predictions. Gated on eval_mode -- in "fit" mode a
    #    prediction node evaluates to the fitted estimator (guide pitfall 12).
    def blend_predictions(preds_1, preds_2, mode):
        if mode == "fit":
            return None
        return 0.5 * np.asarray(preds_1) + 0.5 * np.asarray(preds_2)

    pred = pred_1.skb.apply_func(blend_predictions, pred_2, skrub.eval_mode())

    # 7. Score. No cv= here -- the StratifiedShuffleSplit on mark_as_X drives.
    #    The scorer fits the OptimizedRounder on the scored rows, as the
    #    original's final_rounder does.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring=QWK_SCORER
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(f"Final Validation Performance: {search.results_['mean_test_score'].iloc[0]}")
