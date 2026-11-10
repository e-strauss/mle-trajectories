import copy
import math

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = [
    "Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"
]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    """Learn group medians, then the median of the group-imputed ages."""

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        group_values = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        self.fallback_ = X["Age"].fillna(group_values).median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        group_values = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X.assign(
            Age=X["Age"].fillna(group_values).fillna(self.fallback_)
        )


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X.assign(Ticket=X["Ticket"].map(self.counts_).fillna(1))


class GetXY(TransformerMixin, BaseEstimator):
    """Split only the current outer fold's training rows for early stopping."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        parts = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, d) for n, d in zip(cat_dims, emb_dims)]
        )
        layers = []
        previous_dim = sum(emb_dims) + num_features
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            previous_dim = hidden_dim
        # The original architecture explicitly has two output classes.
        layers.append(nn.Linear(previous_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Only the torch training loop; encoding and scaling are separate nodes."""

    def __init__(
        self, use_mixup=True, loss_type="focal", use_ema=True, seed=42
    ):
        self.use_mixup = use_mixup
        self.loss_type = loss_type
        self.use_ema = use_ema
        self.seed = seed

    def fit(self, X, y):
        set_seed(self.seed)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        cat_values = X[CAT_COLS].to_numpy(dtype=np.int64)
        num_values = X[NUM_COLS].to_numpy(dtype=np.float32)
        cat_dims = (cat_values.max(axis=0) + 1).tolist()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

        model = TabularNeuralNet(
            cat_dims, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)

        if self.use_ema:
            ema_model = copy.deepcopy(model).to(self.device_)
            for parameter in ema_model.parameters():
                parameter.requires_grad = False
            ema_decay = 0.995

        criterion = (
            FocalCrossEntropyLoss(gamma=2.0)
            if self.loss_type == "focal"
            else nn.CrossEntropyLoss()
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-3, weight_decay=1e-4
        )
        t_cat = torch.tensor(
            cat_values, dtype=torch.long, device=self.device_
        )
        t_num = torch.tensor(
            num_values, dtype=torch.float32, device=self.device_
        )
        t_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )

        batch_size = 32
        num_epochs = 150
        num_samples = t_cat.size(0)
        total_steps = num_epochs * math.ceil(num_samples / batch_size)
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda
        )
        mixup_alpha = 0.2 if self.use_mixup else 0.0

        for epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                indices = permutation[start : start + batch_size]
                batch_cat = t_cat[indices]
                batch_num = t_num[indices]
                batch_y = t_y[indices]
                current_size = batch_cat.size(0)

                if mixup_alpha > 0 and current_size > 1:
                    lam = float(
                        torch.distributions.Beta(
                            mixup_alpha, mixup_alpha
                        ).sample().item()
                    )
                    mix_permutation = torch.randperm(current_size)
                    mixed_num = (
                        lam * batch_num
                        + (1.0 - lam) * batch_num[mix_permutation]
                    )
                    cat_mask = (
                        torch.rand(batch_cat.shape, device=self.device_) < lam
                    )
                    mixed_cat = torch.where(
                        cat_mask, batch_cat, batch_cat[mix_permutation]
                    )
                    optimizer.zero_grad()
                    outputs = model(mixed_cat, mixed_num)
                    loss = (
                        lam * criterion(outputs, batch_y)
                        + (1.0 - lam)
                        * criterion(outputs, batch_y[mix_permutation])
                    )
                else:
                    optimizer.zero_grad()
                    outputs = model(batch_cat, batch_num)
                    loss = criterion(outputs, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()

                if self.use_ema:
                    with torch.no_grad():
                        for ema_parameter, parameter in zip(
                            ema_model.parameters(), model.parameters()
                        ):
                            ema_parameter.data.mul_(ema_decay).add_(
                                parameter.data, alpha=1.0 - ema_decay
                            )

        # Preserve the original's parameter-only EMA, including its untouched
        # BatchNorm buffers; do not replace it with a different EMA algorithm.
        self.model_ = ema_model if self.use_ema else model
        self.model_.eval()
        return self

    def predict_proba(self, X):
        t_cat = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        t_num = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        with torch.no_grad():
            return torch.softmax(
                self.model_(t_cat, t_num), dim=1
            ).cpu().numpy()

    def predict(self, X):
        # This intermediate prediction is deliberately a probability, not a
        # class: the final recorded node performs the original ensemble threshold.
        return self.predict_proba(X)[:, 1]


def ensemble_predictions(cb_probability, nn_probability, mode):
    if mode == "fit":
        return None
    cb_probability = np.asarray(cb_probability, dtype=float)[:, 1]
    return (
        0.5 * cb_probability + 0.5 * nn_probability >= 0.5
    ).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the read; importing this file reads nothing.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target. StratifiedShuffleSplit reproduces
    #    the original stratified train_test_split, including its row indices.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Stateful imputations and ticket counts
    #    now learn from training rows only, fixing the original full-table leaks.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = raw_title.map(
        {
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
        }
    ).fillna("Rare")
    title = title.where(raw_title.notna(), "Missing")
    features = X.assign(Title=title)

    features = features.skb.apply(
        GroupAgeImputer(), cols=["Pclass", "Sex", "Age"]
    )
    features = features.assign(
        Sex_Pclass=features["Sex"].astype(str)
        + "_"
        + features["Pclass"].astype(str)
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    # The ordered where chain also preserves the original's NaN -> LargeFamily.
    family_tier = (
        features["FamilySize"]
        .where(features["FamilySize"] <= 4, "LargeFamily")
        .where(~(features["FamilySize"] <= 4), "SmallFamily")
        .where(features["FamilySize"] != 1, "IsAlone")
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(
        HasCabin=features["Cabin"].notna().astype(int)
    )
    ticket_frequency = features[["Ticket"]].skb.apply(TicketFrequency())
    features = features.assign(TicketFreq=ticket_frequency["Ticket"])
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing")
    )
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Fare"]
    )
    features = features[CAT_COLS + NUM_COLS]

    categorical = features[CAT_COLS].astype(str)
    cb_features = categorical.skb.concat([features[NUM_COLS]], axis=1)

    # Keep early stopping and every CatBoost parameter. The original stopped on
    # the scored hold-out, which leaks; this eval set instead comes exclusively
    # from the outer training rows, using the same 20% stratified split settings.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    cb_probability = cb_parts["X"].skb.apply(
        CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_parts.get("y", y),
        fit_kwargs={
            "cat_features": CAT_COLS,
            "eval_set": (cb_parts["X_val"], cb_parts["y_val"]),
            "early_stopping_rounds": 50,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )

    # Sorted ordinal categories reproduce per-column LabelEncoder ordering.
    # Unknown validation categories become 0, exactly as in the original.
    nn_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    ).clip(lower=0).astype(np.int64)
    nn_numeric = (
        features[NUM_COLS]
        .astype(np.float32)
        .skb.apply(StandardScaler())
    )
    nn_features = nn_categorical.skb.concat([nn_numeric], axis=1)
    nn_features = nn_features[CAT_COLS + NUM_COLS]

    # Exactly the four original experiments; no independent choices that would
    # introduce untested combinations. The shared CatBoost branch is unchanged.
    experiments = {
        "Baseline (MixUp + Focal Loss + EMA)": {
            "use_mixup": True, "loss_type": "focal", "use_ema": True
        },
        "Ablation 1 (No Tabular MixUp)": {
            "use_mixup": False, "loss_type": "focal", "use_ema": True
        },
        "Ablation 2 (CrossEntropy Loss vs Focal)": {
            "use_mixup": True, "loss_type": "ce", "use_ema": True
        },
        "Ablation 3 (No EMA Shadow Model)": {
            "use_mixup": True, "loss_type": "focal", "use_ema": False
        },
    }
    variants = {}
    for experiment_name, parameters in experiments.items():
        nn_probability = nn_features.skb.apply(
            NeuralPositiveProbability(**parameters, seed=42), y=y
        )
        variants[experiment_name] = cb_probability.skb.apply_func(
            ensemble_predictions, nn_probability, skrub.eval_mode()
        )
    pred = skrub.choose_from(variants, name="variant").as_data_op()

    # 4. Score every original variant in one search. The splitter on mark_as_X
    #    drives validation; no outer training loop or submission work is needed.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
