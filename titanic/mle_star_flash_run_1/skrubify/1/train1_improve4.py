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
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    """Learn grouped medians and the subsequent global fallback."""

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        self.fallback_ = self._group_fill(X).median()
        return self

    def _group_fill(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        medians = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X["Age"].fillna(medians)

    def transform(self, X):
        return self._group_fill(X).fillna(self.fallback_).to_frame("Age")


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).fillna(1).to_frame("TicketFreq")


class FareMedianImputer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.median_ = X["Fare"].median()
        return self

    def transform(self, X):
        return X["Fare"].fillna(self.median_).to_frame("Fare")


class GetXY(TransformerMixin, BaseEstimator):
    """Reserve an early-stopping set from this outer fold's training rows."""

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
        prev_dim = sum(emb_dims) + num_features
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
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


class EMANeuralProbability(ClassifierMixin, BaseEstimator):
    """The original torch training loop; preprocessing remains in the graph."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
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
        self.ema_model_ = copy.deepcopy(model).to(self.device_)
        for parameter in self.ema_model_.parameters():
            parameter.requires_grad = False

        criterion = FocalCrossEntropyLoss(gamma=2.0)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-3, weight_decay=1e-4
        )
        t_cat = torch.tensor(
            cat_values, dtype=torch.long, device=self.device_
        )
        t_num = torch.tensor(
            num_values, dtype=torch.float32, device=self.device_
        )
        # Tensor conversion is internal to the model, not a target DataOp transform.
        t_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )

        batch_size = 32
        num_epochs = 150
        num_samples = t_cat.size(0)
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

        for epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(num_samples)

            for i in range(0, num_samples, batch_size):
                batch_indices = permutation[i : i + batch_size]
                batch_cat = t_cat[batch_indices]
                batch_num = t_num[batch_indices]
                batch_y = t_y[batch_indices]
                current_batch_size = batch_cat.size(0)

                if mixup_alpha > 0 and current_batch_size > 1:
                    lam = float(
                        torch.distributions.Beta(
                            mixup_alpha, mixup_alpha
                        ).sample().item()
                    )
                    mix_perm = torch.randperm(current_batch_size)
                    mixed_num = (
                        lam * batch_num
                        + (1.0 - lam) * batch_num[mix_perm]
                    )
                    cat_mask = (
                        torch.rand(batch_cat.shape, device=self.device_) < lam
                    )
                    mixed_cat = torch.where(
                        cat_mask, batch_cat, batch_cat[mix_perm]
                    )
                    optimizer.zero_grad()
                    outputs = model(mixed_cat, mixed_num)
                    loss = (
                        lam * criterion(outputs, batch_y)
                        + (1.0 - lam)
                        * criterion(outputs, batch_y[mix_perm])
                    )
                else:
                    optimizer.zero_grad()
                    outputs = model(batch_cat, batch_num)
                    loss = criterion(outputs, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    for ema_parameter, parameter in zip(
                        self.ema_model_.parameters(), model.parameters()
                    ):
                        ema_parameter.data.mul_(ema_decay).add_(
                            parameter.data, alpha=1.0 - ema_decay
                        )

            # The original validation pass only updated an unused best_val_acc.
            # It neither stopped training nor restored a checkpoint. Retain the
            # final EMA weights after all 150 epochs, as the original does.
            self.ema_model_.eval()

        return self

    def predict(self, X):
        cat_tensor = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        num_tensor = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        self.ema_model_.eval()
        with torch.no_grad():
            logits = self.ema_model_(cat_tensor, num_tensor)
            return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def ensemble_predictions(cb_probability, nn_probability, mode):
    # prediction_type="Probability" is (n_samples, n_classes); column 1 is P(1).
    if mode == "fit":
        return None
    cb_probability = np.asarray(cb_probability, dtype=float)[:, 1]
    return (
        0.5 * cb_probability + 0.5 * nn_probability >= 0.5
    ).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records, but does not execute, the CSV read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — raw target and the exact stratified 80/20 outer hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and independent learned preprocessing.
    # These statistics were originally computed before splitting; learn them
    # on training rows only to remove that leakage.
    age = X[["Pclass", "Sex", "Age"]].skb.apply(GroupAgeImputer())
    fare = X[["Fare"]].skb.apply(FareMedianImputer())
    ticket_frequency = X[["Ticket"]].skb.apply(TicketFrequency())

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

    features = X.assign(Age=age["Age"])
    features = features.assign(Title=title)
    features = features.assign(
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str)
    )
    features = features.assign(FamilySize=X["SibSp"] + X["Parch"] + 1)
    family_size = features["FamilySize"]
    # Start with string dtype so all replacements remain string-valued.
    family_tier = family_size.astype(str).where(
        family_size == 1, "SmallFamily"
    )
    family_tier = family_tier.where(family_size != 1, "IsAlone")
    family_tier = family_tier.where(
        (family_size == 1) | (family_size <= 4), "LargeFamily"
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=X["Cabin"].notna().astype(int))
    features = features.assign(TicketFreq=ticket_frequency["TicketFreq"])
    features = features.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.assign(Fare=fare["Fare"])
    features = features[FEATURE_COLS]

    categorical = features[CAT_COLS].astype(str)
    numerical = features[NUM_COLS]
    cb_features = categorical.skb.concat([numerical], axis=1)[FEATURE_COLS]

    # CatBoost originally early-stopped on the scored hold-out. Keep patience
    # and iteration limit, but reserve an internal stratified 20% eval set from
    # the outer training rows. The original leaky score is not reproduced.
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

    # Sorted string vocabularies match LabelEncoder. OrdinalEncoder requires
    # an unknown sentinel outside its seen codes: use -1 during encoding, then
    # replace it with 0 to reproduce the original unseen-category fallback.
    nn_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    ).replace(-1, 0)
    nn_numerical = numerical.astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_categorical.skb.concat(
        [nn_numerical], axis=1
    )[FEATURE_COLS]
    nn_probability = nn_features.skb.apply(EMANeuralProbability(), y=y)

    # Two components of ONE original ensemble, not separately scored variants.
    # Preserve positive-column averaging and the inclusive 0.5 threshold.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 4. Score — the splitter on mark_as_X owns the outer validation.
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
