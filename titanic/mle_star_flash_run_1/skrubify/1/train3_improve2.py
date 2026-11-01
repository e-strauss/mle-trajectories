"""train3_improve2: isotonic calibration and a power-mean blend on the hold-out.

Features, the split, CatBoost, and the neural net are the matching ``train3``
plan. The original calibrates the hold-out probabilities with five isotonic
folds, then Nelder-Mead picks the weight, power, and threshold on those rows.
``PowerMeanBlend`` repeats that and returns the resulting labels in the scored
fold's row order.
"""

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS


def _positive_probability(model, X):
    return np.asarray(model.predict_proba(X))[:, 1]


def _power_mean(w1, w2, power, p1, p2):
    total = w1 + w2 + 1e-8
    w1_norm, w2_norm = w1 / total, w2 / total
    if abs(power) < 1e-3:
        return np.exp(w1_norm * np.log(p1) + w2_norm * np.log(p2))
    return np.power(
        w1_norm * np.power(p1, power) + w2_norm * np.power(p2, power),
        1.0 / power,
    )


class PowerMeanBlend(ClassifierMixin, BaseEstimator):
    """Isotonic out-of-fold calibration, then the original Nelder-Mead blend."""

    def fit(self, X, y, cb_model=None, nn_model=None, X_val=None, y_val=None):
        self.classes_ = np.unique(np.asarray(y))
        labels = np.asarray(y_val).ravel()
        cb_probs = _positive_probability(cb_model, X_val)
        nn_probs = _positive_probability(nn_model, X_val)
        cb_cal = np.zeros_like(cb_probs, dtype=float)
        nn_cal = np.zeros_like(nn_probs, dtype=float)
        folds = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, val_index in folds.split(cb_probs, labels):
            iso_cb = IsotonicRegression(out_of_bounds="clip")
            iso_nn = IsotonicRegression(out_of_bounds="clip")
            iso_cb.fit(cb_probs[train_index], labels[train_index])
            iso_nn.fit(nn_probs[train_index], labels[train_index])
            cb_cal[val_index] = iso_cb.predict(cb_probs[val_index])
            nn_cal[val_index] = iso_nn.predict(nn_probs[val_index])
        cb_cal = np.clip(cb_cal, 1e-6, 1.0 - 1e-6)
        nn_cal = np.clip(nn_cal, 1e-6, 1.0 - 1e-6)

        def objective(params):
            weight, power, threshold = params
            weight = np.clip(weight, 0.0, 1.0)
            threshold = np.clip(threshold, 0.01, 0.99)
            blended = _power_mean(weight, 1.0 - weight, power, cb_cal, nn_cal)
            return -accuracy_score(labels, (blended >= threshold).astype(int))

        result = minimize(
            objective,
            x0=[0.5, 1.0, 0.5],
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
        )
        weight, power, threshold = result.x
        weight = np.clip(weight, 0.0, 1.0)
        threshold = np.clip(threshold, 0.01, 0.99)
        blended = _power_mean(weight, 1.0 - weight, power, cb_cal, nn_cal)
        predictions = (blended >= threshold).astype(int)
        self.predictions_ = pd.Series(predictions, index=X_val.index)
        return self

    def predict(self, X):
        return self.predictions_.loc[X.index].to_numpy()


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


class TorchTabularClassifier(ClassifierMixin, BaseEstimator):
    """The original torch loop, including its train-only encoders."""

    def __init__(
        self,
        cat_cols=tuple(CAT_COLS),
        num_cols=tuple(NUM_COLS),
        hidden_dims=(64, 32),
        batch_size=32,
        num_epochs=150,
        learning_rate=1e-2,
        weight_decay=1e-4,
        random_state=42,
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _encode_cats(self, X, fit):
        arrays = []
        if fit:
            self.label_encoders_ = {}
        for column in self.cat_cols:
            values = X[column].astype(str)
            if fit:
                encoder = LabelEncoder().fit(values)
                self.label_encoders_[column] = encoder
                arrays.append(encoder.transform(values))
            else:
                encoder = self.label_encoders_[column]
                mapping = {
                    value: index for index, value in enumerate(encoder.classes_)
                }
                arrays.append(
                    np.asarray(
                        [mapping.get(value, 0) for value in values.to_numpy()],
                        dtype=np.int64,
                    )
                )
        return np.stack(arrays, axis=1).astype(np.int64)

    def _encode_nums(self, X, fit):
        raw = X.loc[:, list(self.num_cols)].to_numpy(dtype=np.float32)
        if fit:
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(raw)
        return self.scaler_.transform(raw)

    def fit(self, X, y, X_val=None, y_val=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(np.asarray(y))
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        X_cat = self._encode_cats(X, fit=True)
        X_num = self._encode_nums(X, fit=True)
        X_cat_val = self._encode_cats(X_val, fit=False)
        X_num_val = self._encode_nums(X_val, fit=False)
        cat_dims = [
            len(self.label_encoders_[column].classes_) for column in self.cat_cols
        ]
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

        self.model_ = TabularNeuralNet(
            cat_dims,
            emb_dims,
            num_features=X_num.shape[1],
            hidden_dims=self.hidden_dims,
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        x_cat = torch.tensor(X_cat, dtype=torch.long, device=self.device_)
        x_num = torch.tensor(X_num, dtype=torch.float32, device=self.device_)
        target = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        v_cat = torch.tensor(X_cat_val, dtype=torch.long, device=self.device_)
        v_num = torch.tensor(X_num_val, dtype=torch.float32, device=self.device_)
        y_validation = np.asarray(y_val)

        for _ in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(x_cat.size(0))

            for start in range(0, x_cat.size(0), self.batch_size):
                batch = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(x_cat[batch], x_num[batch])
                loss = criterion(outputs, target[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(v_cat, v_num)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_validation, predictions))

        return self

    def predict_proba(self, X):
        x_cat = torch.tensor(
            self._encode_cats(X, fit=False), dtype=torch.long, device=self.device_
        )
        x_num = torch.tensor(
            self._encode_nums(X, fit=False),
            dtype=torch.float32,
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(x_cat, x_num)
            return torch.softmax(outputs, dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


with skrub.config_context(eager_data_ops=False):
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Full-table statistics, matching prepare_features before its split.
    raw_title = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
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
    age_median = data.groupby(["Pclass", "Sex"])["Age"].transform("median")
    age = data["Age"].fillna(age_median).fillna(data["Age"].median())
    family_size = data["SibSp"] + data["Parch"] + 1
    family_tier = (
        family_size
        .where(family_size == 1, "SmallFamily")
        .where(family_size <= 4, "LargeFamily")
        .where(family_size != 1, "IsAlone")
    )
    ticket_frequency = data["Ticket"].map(data["Ticket"].value_counts()).fillna(1)
    prepared = data.assign(
        Title=title,
        Age=age,
        Sex_Pclass=data["Sex"].astype(str) + "_" + data["Pclass"].astype(str),
        FamilySize=family_size,
        FamilyTier=family_tier,
        HasCabin=data["Cabin"].notna().astype(int),
        TicketFreq=ticket_frequency,
        Embarked=data["Embarked"].fillna("Missing"),
        Fare=data["Fare"].fillna(data["Fare"].median()),
    )
    model_features = prepared[FEATURE_COLS]
    for column in CAT_COLS:
        model_features = model_features.assign(
            **{column: model_features[column].astype(str)}
        )

    # Same hold-out the original scores and early-stops on. The marks below
    # are separate nodes, so this split stays the full table.
    split_target = data["Survived"]
    fixed_split = model_features.skb.apply_func(
        train_test_split,
        split_target,
        test_size=0.2,
        random_state=42,
        stratify=split_target,
    )
    X_validation = fixed_split[1]
    y_validation = fixed_split[3]

    y = data["Survived"].skb.mark_as_y()
    X = model_features.skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    cb_probability = X.skb.apply(
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
            "eval_set": (X_validation, y_validation),
            "early_stopping_rounds": 50,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    nn_probability = X.skb.apply(
        TorchTabularClassifier(),
        y=y,
        fit_kwargs={"X_val": X_validation, "y_val": y_validation},
    )
    pred = X.skb.apply(
        PowerMeanBlend(),
        y=y,
        fit_kwargs={
            "cb_model": cb_probability.skb.applied_estimator,
            "nn_model": nn_probability.skb.applied_estimator,
            "X_val": X_validation,
            "y_val": y_validation,
        },
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
