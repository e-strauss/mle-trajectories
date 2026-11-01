import copy
import math

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS
TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}


def engineer(data):
    """Full-table features, including the rows the original later scores."""
    raw_title = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = (
        raw_title.map(TITLE_MAP).fillna("Rare").where(raw_title.notna(), "Missing")
    )
    features = data.assign(Title=title)
    grouped_age = features.groupby(["Pclass", "Sex"])["Age"].transform("median")
    features = features.assign(Age=features["Age"].fillna(grouped_age))
    features = features.assign(Age=features["Age"].fillna(features["Age"].median()))
    features = features.assign(
        Sex_Pclass=features["Sex"].astype(str) + "_" + features["Pclass"].astype(str)
    )
    features = features.assign(FamilySize=features["SibSp"] + features["Parch"] + 1)
    family_size = features["FamilySize"]
    family_tier = (
        family_size.le(4)
        .map({True: "SmallFamily", False: "LargeFamily"})
        .where(family_size.ne(1), "IsAlone")
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=features["Cabin"].notna().astype(int))
    ticket_counts = features["Ticket"].value_counts()
    features = features.assign(
        TicketFreq=features["Ticket"].map(ticket_counts).fillna(1)
    )
    features = features.assign(Embarked=features["Embarked"].fillna("Missing"))
    features = features.assign(
        Fare=features["Fare"].fillna(features["Fare"].median())
    )
    return features


def as_model_frame(frame):
    """Feature columns in the original order, with categoricals as strings."""
    selected = frame[FEATURE_COLS]
    return selected.assign(
        **{column: selected[column].astype(str) for column in CAT_COLS}
    )


def holdout_rows(df):
    indices = np.arange(len(df))
    _, idx_val, _, _ = train_test_split(
        indices,
        df["Survived"],
        test_size=0.2,
        random_state=42,
        stratify=df["Survived"],
    )
    return df.iloc[idx_val]


def fit_encoders(X):
    encoders = [
        LabelEncoder().fit(X[column].astype(str)) for column in CAT_COLS
    ]
    scaler = StandardScaler().fit(X[NUM_COLS].to_numpy().astype(np.float32))
    return encoders, scaler


def encode_matrices(X, encoders, scaler):
    columns = []
    for column, encoder in zip(CAT_COLS, encoders):
        mapping = {value: code for code, value in enumerate(encoder.classes_)}
        encoded = np.array(
            [mapping.get(value, 0) for value in X[column].astype(str).to_numpy()]
        )
        columns.append(encoded)
    categorical = np.stack(columns, axis=1).astype(np.int64)
    numeric = scaler.transform(X[NUM_COLS].to_numpy().astype(np.float32))
    return categorical, numeric


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_c, e_dim)
                for num_c, e_dim in zip(cat_dims, emb_dims)
            ]
        )
        layers = []
        prev_dim = sum(emb_dims) + num_features
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.network(torch.cat(embs + [x_num], dim=1))


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class EMATabularClassifier(ClassifierMixin, BaseEstimator):
    """EMA network. The per-epoch accuracy on the scored hold-out is recorded
    and does not change the weights, matching the original.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y, *, X_val, y_val):
        y = np.asarray(y).ravel()
        y_val = np.asarray(y_val).ravel()
        self.classes_ = np.unique(y)
        self.encoders_, self.scaler_ = fit_encoders(X)
        train_cat, train_num = encode_matrices(X, self.encoders_, self.scaler_)
        val_cat, val_num = encode_matrices(X_val, self.encoders_, self.scaler_)

        torch.manual_seed(self.random_state)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dimensions = [len(encoder.classes_) for encoder in self.encoders_]
        embedding_dimensions = [
            max(2, int(round(dimension**0.5 * 2))) for dimension in dimensions
        ]
        model = TabularNeuralNet(
            dimensions, embedding_dimensions, num_features=len(NUM_COLS)
        ).to(self.device_)
        ema_model = copy.deepcopy(model).to(self.device_)
        for param in ema_model.parameters():
            param.requires_grad = False

        criterion = FocalCrossEntropyLoss(gamma=2.0)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-3, weight_decay=1e-4
        )
        train_cat_t = torch.tensor(
            train_cat, dtype=torch.long, device=self.device_
        )
        train_num_t = torch.tensor(
            train_num, dtype=torch.float32, device=self.device_
        )
        train_y = torch.tensor(y, dtype=torch.long, device=self.device_)
        val_cat_t = torch.tensor(val_cat, dtype=torch.long, device=self.device_)
        val_num_t = torch.tensor(
            val_num, dtype=torch.float32, device=self.device_
        )

        batch_size = 32
        num_epochs = 150
        num_samples = train_cat_t.size(0)
        total_steps = num_epochs * math.ceil(num_samples / batch_size)
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda
        )
        mixup_alpha = 0.2
        ema_decay = 0.995
        best_val_acc = 0.0

        for _epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                batch = permutation[start : start + batch_size]
                batch_cat = train_cat_t[batch]
                batch_num = train_num_t[batch]
                batch_y = train_y[batch]
                cur_bs = batch_cat.size(0)

                if mixup_alpha > 0 and cur_bs > 1:
                    lam = float(
                        torch.distributions.Beta(mixup_alpha, mixup_alpha)
                        .sample()
                        .item()
                    )
                    mix_perm = torch.randperm(cur_bs)
                    batch_num_mixed = (
                        lam * batch_num + (1.0 - lam) * batch_num[mix_perm]
                    )
                    cat_mask = (
                        torch.rand(batch_cat.shape, device=self.device_) < lam
                    )
                    batch_cat_mixed = torch.where(
                        cat_mask, batch_cat, batch_cat[mix_perm]
                    )
                    optimizer.zero_grad()
                    outputs = model(batch_cat_mixed, batch_num_mixed)
                    loss = lam * criterion(outputs, batch_y) + (
                        1.0 - lam
                    ) * criterion(outputs, batch_y[mix_perm])
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
                with torch.no_grad():
                    for ema_p, p in zip(
                        ema_model.parameters(), model.parameters()
                    ):
                        ema_p.data.mul_(ema_decay).add_(
                            p.data, alpha=1.0 - ema_decay
                        )

            ema_model.eval()
            with torch.no_grad():
                val_outputs = ema_model(val_cat_t, val_num_t)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_preds)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

        self.model_ = ema_model.eval()
        return self

    def predict(self, X):
        categorical, numeric = encode_matrices(X, self.encoders_, self.scaler_)
        x_cat = torch.tensor(
            categorical, dtype=torch.long, device=self.device_
        )
        x_num = torch.tensor(numeric, dtype=torch.float32, device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(x_cat, x_num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def _positive_class_probability(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        return array[:, 1]
    return array.reshape(-1)


class MeanProbabilityEnsemble(ClassifierMixin, BaseEstimator):
    """Threshold the equal-weight average of two positive-class probabilities."""

    def fit(self, X, y, nn_probability=None):
        cb_probability = _positive_class_probability(X)
        nn_probability = _positive_class_probability(nn_probability)
        if cb_probability.shape != nn_probability.shape:
            raise ValueError(
                "CatBoost and neural-net probabilities have different lengths: "
                f"{cb_probability.shape[0]} and {nn_probability.shape[0]}."
            )
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict(self, X, nn_probability=None):
        cb_probability = _positive_class_probability(X)
        nn_probability = _positive_class_probability(nn_probability)
        averaged = 0.5 * cb_probability + 0.5 * nn_probability
        return (averaged >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)
    features = engineer(data)

    holdout = features.skb.apply_func(holdout_rows)
    X_val = as_model_frame(holdout)
    y_val = holdout["Survived"]

    y = features["Survived"].skb.mark_as_y()
    X = features.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )
    X_model = as_model_frame(X)

    cb_probability = X_model.skb.apply(
        CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=y,
        fit_kwargs={
            "cat_features": CAT_COLS,
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 50,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    nn_probability = X_model.skb.apply(
        EMATabularClassifier(),
        y=y,
        fit_kwargs={"X_val": X_val, "y_val": y_val},
    )
    pred = cb_probability.skb.apply(
        MeanProbabilityEnsemble(),
        y=y,
        fit_kwargs={"nn_probability": nn_probability},
        predict_kwargs={"nn_probability": nn_probability},
    )

    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
