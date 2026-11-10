import copy

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
    """Learn the original grouped medians and subsequent global fallback."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        grouped_values = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        self.fallback_ = X["Age"].fillna(grouped_values).median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        grouped_values = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        age = X["Age"].fillna(grouped_values).fillna(self.fallback_)
        return age.to_frame("Age")


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).fillna(1).to_frame("TicketFreq")


class GetXY(TransformerMixin, BaseEstimator):
    """Reserve an eval set from this outer fold's training rows only."""

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
        # The original network explicitly has two output classes.
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class SnapshotNeuralNet(ClassifierMixin, BaseEstimator):
    """Only the torch training loop and five-seed snapshot ensemble live here."""

    def __init__(
        self,
        seeds=(42, 123, 456, 789, 2024),
        num_epochs=100,
        batch_size=32,
        warmup_epochs=25,
    ):
        self.seeds = seeds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs

    def _tensors(self, X):
        cats = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        nums = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        return cats, nums

    def fit(self, X, y, eval_set=None, cat_dims=None):
        self.classes_ = np.unique(y)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cat_dims_ = np.asarray(cat_dims, dtype=int).tolist()
        self.emb_dims_ = [
            max(2, int(round(dim**0.5 * 2))) for dim in self.cat_dims_
        ]

        X_val, y_val = eval_set
        x_cat, x_num = self._tensors(X)
        val_cat, val_num = self._tensors(X_val)
        target = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        val_target = torch.tensor(
            np.asarray(y_val), dtype=torch.long, device=self.device_
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.seed_snapshots_ = []

        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = TabularNeuralNet(
                self.cat_dims_, self.emb_dims_, num_features=len(NUM_COLS)
            ).to(self.device_)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-3, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=25, T_mult=1, eta_min=1e-5
            )

            snapshots = []
            best_val_loss = float("inf")
            best_state = None

            for epoch in range(self.num_epochs):
                model.train()
                permutation = torch.randperm(x_cat.size(0))

                for start in range(0, x_cat.size(0), self.batch_size):
                    indices = permutation[start : start + self.batch_size]
                    optimizer.zero_grad()
                    outputs = model(x_cat[indices], x_num[indices])
                    loss = criterion(outputs, target[indices])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()

                scheduler.step()
                model.eval()
                with torch.no_grad():
                    val_outputs = model(val_cat, val_num)
                    val_loss = criterion(val_outputs, val_target).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())

                # Preserve the original update/comparison order, including <=.
                if epoch >= self.warmup_epochs and (
                    (epoch + 1) % 25 == 0 or val_loss <= best_val_loss
                ):
                    snapshots.append(copy.deepcopy(model.state_dict()))

            if not snapshots:
                snapshots = [best_state]
            self.seed_snapshots_.append(snapshots[-4:])

        return self

    def predict(self, X):
        # Return positive-class probabilities; the plan performs the final
        # 50/50 blend and >= 0.5 threshold, exactly as in the original.
        x_cat, x_num = self._tensors(X)
        all_seed_probs = []
        model = TabularNeuralNet(
            self.cat_dims_, self.emb_dims_, num_features=len(NUM_COLS)
        ).to(self.device_)

        for snapshots in self.seed_snapshots_:
            snapshot_probs = []
            for state in snapshots:
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    outputs = model(x_cat, x_num)
                    snapshot_probs.append(
                        torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    )
            all_seed_probs.append(np.mean(snapshot_probs, axis=0))

        return np.mean(all_seed_probs, axis=0)


def blend_predictions(cb_probs, nn_probs, mode):
    # Predictor nodes contain estimators rather than predictions in fit mode.
    # prediction_type="Probability" is (n_samples, n_classes); column 1 is P(1).
    if mode == "fit":
        return None
    cb_probs = np.asarray(cb_probs, dtype=float)[:, 1]
    return (0.5 * cb_probs + 0.5 * nn_probs >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the read; importing this module performs no IO.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and features. StratifiedShuffleSplit reproduces the
    #    original single train_test_split(..., stratify=y) exactly.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Learned statistics are now fitted only
    #    on outer-training rows: the original full-table imputations and ticket
    #    counts leaked information from the scored hold-out.
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

    age = X[["Pclass", "Sex", "Age"]].skb.apply(
        GroupedAgeImputer(), no_wrap=True
    )
    ticket_freq = X[["Ticket"]].skb.apply(TicketFrequency(), no_wrap=True)
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))

    features = X.assign(Title=title)
    features = features.assign(Age=age["Age"])
    features = features.assign(
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str)
    )
    features = features.assign(FamilySize=X["SibSp"] + X["Parch"] + 1)
    family_size = features["FamilySize"]
    family_tier = (
        family_size.astype(str)
        .where(family_size == 1, "SmallFamily")
        .where(family_size <= 4, "LargeFamily")
        .where(family_size != 1, "IsAlone")
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=X["Cabin"].notna().astype(int))
    features = features.assign(TicketFreq=ticket_freq["TicketFreq"])
    features = features.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.assign(Fare=fare["Fare"])

    categorical = features[CAT_COLS].astype(str)
    numeric = features[NUM_COLS]
    cb_features = categorical.skb.concat([numeric], axis=1)

    # OrdinalEncoder on strings uses the same sorted vocabularies as the
    # original LabelEncoders. Unknown categories map to index zero.
    encoded_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    ).replace(-1, 0)
    scaled_numeric = numeric.astype(np.float32).skb.apply(StandardScaler())
    nn_features = encoded_categorical.skb.concat([scaled_numeric], axis=1)

    # Include all outer-training vocabulary entries.
    cat_dims = encoded_categorical.max().astype(int) + 1

    # 4. Both models originally selected checkpoints using the scored hold-out.
    #    Keep CatBoost's patience and the NN's snapshot-selection logic, but use
    #    an internal stratified 20% eval set drawn from outer-training rows.
    #    This fixes that leakage, so the original leaky score is not reproduced.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)

    cb_probs = cb_parts["X"].skb.apply(
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
    nn_probs = nn_parts["X"].skb.apply(
        SnapshotNeuralNet(),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "eval_set": (nn_parts["X_val"], nn_parts["y_val"]),
            "cat_dims": cat_dims,
        },
    )
    pred = cb_probs.skb.apply_func(
        blend_predictions, nn_probs, skrub.eval_mode()
    )

    # 5. Score the one original ensemble, not its individual components.
    #    No cv= here: the marked splitter supplies the original hold-out rows.
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
