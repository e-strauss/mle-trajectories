import copy

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    """Learn Pclass/Sex age medians; global fallback is a separate node."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        replacements = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X.assign(Age=X["Age"].fillna(replacements))


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X.assign(TicketFreq=X["Ticket"].map(self.counts_).fillna(1))


class GetXY(TransformerMixin, BaseEstimator):
    """Reserve validation rows from this outer fold's training rows only."""

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
            [nn.Embedding(n, e) for n, e in zip(cat_dims, emb_dims)]
        )
        layers = []
        previous = sum(emb_dims) + num_features
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class NeuralCheckpointEnsemble(ClassifierMixin, BaseEstimator):
    """Original five-seed torch training and top-three weight averaging.

    Encoding, scaling, and validation-row selection remain visible in the plan.
    Only model training and checkpoint handling are encapsulated here.
    """

    def __init__(
        self,
        seeds=(42, 1337, 2024, 777, 999),
        num_epochs=150,
        batch_size=32,
        top_k_checkpoints=3,
    ):
        self.seeds = seeds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.top_k_checkpoints = top_k_checkpoints

    def fit(self, X, y, eval_set=None):
        X_val, y_val = eval_set
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        cat_train = X[CAT_COLS].to_numpy(dtype=np.int64)
        cat_val = X_val[CAT_COLS].to_numpy(dtype=np.int64)
        # Encoding precedes the inner split, so the union of these partitions
        # recovers the encoder's complete outer-training vocabulary dimensions.
        self.cat_dims_ = (
            np.maximum(cat_train.max(axis=0), cat_val.max(axis=0)) + 1
        ).tolist()
        self.emb_dims_ = [
            max(2, int(round(dim**0.5 * 2))) for dim in self.cat_dims_
        ]

        t_cat_train = torch.tensor(
            cat_train, dtype=torch.long, device=self.device_
        )
        t_num_train = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        t_y_train = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        t_cat_val = torch.tensor(
            cat_val, dtype=torch.long, device=self.device_
        )
        t_num_val = torch.tensor(
            X_val[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        t_y_val = torch.tensor(
            np.asarray(y_val), dtype=torch.long, device=self.device_
        )

        self.models_ = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = TabularNeuralNet(
                self.cat_dims_, self.emb_dims_, len(NUM_COLS)
            ).to(self.device_)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-2, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=10
            )
            best_checkpoints = []

            for epoch in range(self.num_epochs):
                model.train()
                permutation = torch.randperm(t_cat_train.size(0))

                for start in range(0, t_cat_train.size(0), self.batch_size):
                    batch_indices = permutation[start : start + self.batch_size]
                    batch_cat = t_cat_train[batch_indices]
                    batch_num = t_num_train[batch_indices]
                    batch_y = t_y_train[batch_indices]

                    if batch_num.size(0) > 1 and np.random.rand() < 0.5:
                        noisy = batch_num + torch.randn_like(batch_num) * 0.015
                        lam = np.random.beta(0.4, 0.4)
                        permutation_mix = torch.randperm(batch_num.size(0))
                        mixed = (
                            lam * noisy
                            + (1.0 - lam) * noisy[permutation_mix]
                        )
                        optimizer.zero_grad()
                        outputs = model(batch_cat, mixed)
                        loss = (
                            lam * criterion(outputs, batch_y)
                            + (1.0 - lam)
                            * criterion(outputs, batch_y[permutation_mix])
                        )
                    else:
                        noise = torch.randn_like(batch_num) * 0.01
                        optimizer.zero_grad()
                        outputs = model(batch_cat, batch_num + noise)
                        loss = criterion(outputs, batch_y)

                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    outputs = model(t_cat_val, t_num_val)
                    val_loss = criterion(outputs, t_y_val).item()
                    predictions = outputs.argmax(dim=1).cpu().numpy()
                    val_accuracy = accuracy_score(y_val, predictions)

                scheduler.step(val_accuracy)
                best_checkpoints.append(
                    (
                        val_accuracy,
                        -val_loss,
                        copy.deepcopy(model.state_dict()),
                    )
                )
                best_checkpoints.sort(
                    key=lambda checkpoint: (checkpoint[0], checkpoint[1]),
                    reverse=True,
                )
                if len(best_checkpoints) > self.top_k_checkpoints:
                    best_checkpoints.pop()

            averaged_state = copy.deepcopy(best_checkpoints[0][2])
            for key in averaged_state:
                averaged_state[key] = (
                    torch.stack(
                        [
                            checkpoint[2][key].float()
                            for checkpoint in best_checkpoints
                        ],
                        dim=0,
                    )
                    .mean(dim=0)
                    .to(averaged_state[key].dtype)
                )
            model.load_state_dict(averaged_state)
            model.eval()
            self.models_.append(model)

        return self

    def predict(self, X):
        # Intermediate prediction: positive-class probabilities, not labels.
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
        probabilities = []
        with torch.no_grad():
            for model in self.models_:
                outputs = model(t_cat, t_num)
                probabilities.append(
                    torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                )
        return np.mean(probabilities, axis=0)


def ensemble_predictions(cb_probabilities, nn_probabilities, mode):
    # prediction_type="Probability" is (n_samples, n_classes); column 1 is P(1).
    if mode == "fit":
        return None
    cb_probabilities = np.asarray(cb_probabilities, dtype=float)[:, 1]
    return (
        0.5 * cb_probabilities + 0.5 * nn_probabilities >= 0.5
    ).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records, but does not execute, the CSV read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — raw target and the original stratified holdout.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing, feature engineering, and models.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = (
        raw_title.map(
            {
                "Mr": "Mr",
                "Miss": "Miss",
                "Mrs": "Mrs",
                "Master": "Master",
                "Mlle": "Miss",
                "Ms": "Miss",
                "Mme": "Mrs",
            }
        )
        .fillna("Rare")
        .where(raw_title.notna(), "Missing")
    )
    features = X.assign(Title=title)

    # These statistics originally used the entire table. Separate fitted nodes
    # now learn group medians, the fallback median, and ticket counts per fold.
    features = features.skb.apply(GroupAgeImputer())
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Age"]
    )
    features = features.assign(
        Sex_Pclass=features["Sex"].astype(str)
        + "_"
        + features["Pclass"].astype(str)
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    size = features["FamilySize"]
    tier = (
        size.astype(str)
        .where(size != 1, "IsAlone")
        .where(~((size != 1) & (size <= 4)), "SmallFamily")
        .where((size == 1) | (size <= 4), "LargeFamily")
    )
    features = features.assign(FamilyTier=tier)
    features = features.assign(
        HasCabin=features["Cabin"].notna().astype(int)
    )
    features = features.skb.apply(TicketFrequency())
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing")
    )
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Fare"]
    )
    features = features[FEATURE_COLS]

    categorical = features[CAT_COLS].astype(str)
    cb_features = categorical.skb.concat(
        [features[NUM_COLS]], axis=1
    )[FEATURE_COLS]

    # The original early-stopped CatBoost on the scored holdout. Preserve
    # patience and iterations, but reserve 20% of outer-training rows instead.
    # That original leaky validation score cannot be reproduced honestly.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    cb_probabilities = cb_parts["X"].skb.apply(
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

    # OrdinalEncoder reproduces LabelEncoder's sorted per-column codes.
    # Its unknown code must not collide with a fitted category's code: encode
    # unknowns as -1 first, then explicitly restore the original fallback 0.
    nn_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    ).replace(-1, 0)
    nn_numeric = (
        features[NUM_COLS]
        .astype(np.float32)
        .skb.apply(StandardScaler())
    )
    nn_features = nn_categorical.skb.concat(
        [nn_numeric], axis=1
    )[FEATURE_COLS]

    # The original torch scheduler/checkpoint selection also inspected the
    # scored holdout. Use only outer-training rows for that validation now.
    # The recorded inner split matches CatBoost's; torch training, seed
    # ensembling, and checkpoint averaging remain inside the model.
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_probabilities = nn_parts["X"].skb.apply(
        NeuralCheckpointEnsemble(
            seeds=(42, 1337, 2024, 777, 999),
            num_epochs=150,
            batch_size=32,
            top_k_checkpoints=3,
        ),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "eval_set": (nn_parts["X_val"], nn_parts["y_val"])
        },
    )

    # Preserve the original probability arithmetic and inclusive threshold.
    pred = cb_probabilities.skb.apply_func(
        ensemble_predictions, nn_probabilities, skrub.eval_mode()
    )

    # 4. Score — one holdout accuracy for the original ensemble. No cv here:
    # mark_as_X owns the splitter. No test prediction or output files are needed.
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
