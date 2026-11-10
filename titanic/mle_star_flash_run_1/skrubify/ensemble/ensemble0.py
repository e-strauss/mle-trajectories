import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GroupAgeMedian(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        replacement = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X[["Age"]].assign(Age=X["Age"].fillna(replacement))


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {"TicketFreq": X["Ticket"].map(self.counts_).fillna(1)},
            index=X.index,
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Expose the internal validation split as a recorded estimator step.

    In predict mode ``X_val`` is the scored rows: the ensemble members are
    applied to ``X_val``, so the same member node yields holdout probabilities
    while fitting and scored-row probabilities while predicting.
    """

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
        return {"X": X, "X_val": X, "y": None, "y_val": None}


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, d) for n, d in zip(cat_dims, emb_dims)]
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
        embedded = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embedded + [x_num], dim=1))


class TorchTabularClassifier(ClassifierMixin, BaseEstimator):
    """The original torch training loop, without feature preprocessing."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def _tensors(self, X):
        categorical = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        numeric = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        return categorical, numeric

    def fit(self, X, y, X_val=None, y_val=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dimensions = np.asarray(cat_dims, dtype=int).tolist()
        embedding_dimensions = [
            max(2, int(round(dim**0.5 * 2))) for dim in dimensions
        ]
        # Two output classes, as explicitly specified by the original network.
        self.model_ = TabularNeuralNet(
            dimensions, embedding_dimensions, num_features=len(NUM_COLS)
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        train_cat, train_num = self._tensors(X)
        val_cat, val_num = self._tensors(X_val)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )

        for epoch in range(150):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for start in range(0, train_cat.size(0), 32):
                batch = permutation[start : start + 32]
                optimizer.zero_grad()
                outputs = self.model_(train_cat[batch], train_num[batch])
                loss = criterion(outputs, train_y[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(val_cat, val_num)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_val, predictions))

        return self

    def predict_proba(self, X):
        categorical, numeric = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            return (
                torch.softmax(self.model_(categorical, numeric), dim=1)
                .cpu()
                .numpy()
            )

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class FitOn(ClassifierMixin, BaseEstimator):
    """Fit ``estimator`` on the rows passed in fit_kwargs; predict P(class 1) on X.

    The node's input is the blending holdout (scored rows in predict mode), so
    the model's training rows have to arrive through ``X_fit``/``y_fit``.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, X_fit=None, y_fit=None, **fit_params):
        self.estimator_ = clone(self.estimator).fit(X_fit, y_fit, **fit_params)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict_proba(X)[:, 1]


class WeightThresholdSearch(ClassifierMixin, BaseEstimator):
    """The original weight/threshold search over the members' probabilities.

    Columns of X are the CatBoost, XGBoost and NN positive-class probabilities.
    """

    def fit(self, X, y):
        cb_probs, xgb_probs, nn_probs = np.asarray(X, dtype=float).T
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)

        best_score = -1.0
        self.weights_ = (1 / 3, 1 / 3, 1 / 3)
        self.threshold_ = 0.5
        step = 0.05
        thresholds = np.linspace(0.3, 0.7, 41)

        for w1 in np.arange(0.0, 1.0 + step, step):
            for w2 in np.arange(0.0, 1.0 - w1 + step, step):
                w3 = max(0.0, 1.0 - w1 - w2)
                if abs(w1 + w2 + w3 - 1.0) > 1e-5:
                    continue
                probabilities = w1 * cb_probs + w2 * xgb_probs + w3 * nn_probs
                for threshold in thresholds:
                    predictions = (probabilities >= threshold).astype(int)
                    score = accuracy_score(labels, predictions)
                    if score > best_score:
                        best_score = score
                        self.weights_ = (w1, w2, w3)
                        self.threshold_ = threshold
        return self

    def predict(self, X):
        cb_probs, xgb_probs, nn_probs = np.asarray(X, dtype=float).T
        w1, w2, w3 = self.weights_
        probabilities = w1 * cb_probs + w2 * xgb_probs + w3 * nn_probs
        return (probabilities >= self.threshold_).astype(int)


def stack(*columns):
    # A Python function, not np.column_stack itself: stratum turns an unlisted
    # numpy function handed to skrub.deferred into an element-wise op.
    return np.column_stack(columns)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — raw target and the original stratified 80/20 hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Medians and ticket frequencies are
    #    learned on outer-training rows only, fixing full-table leakage.
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

    grouped_age = X[["Pclass", "Sex", "Age"]].skb.apply(GroupAgeMedian())
    age = grouped_age.skb.apply(SimpleImputer(strategy="median"))
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))
    ticket = X[["Ticket"]].skb.apply(TicketFrequency())

    family_size = X["SibSp"] + X["Parch"] + 1
    family_tier = (
        family_size.astype(str)
        .where(family_size <= 4, "LargeFamily")
        .where(family_size > 4, "SmallFamily")
        .where(family_size != 1, "IsAlone")
    )
    features = X.assign(
        Title=title,
        Age=age["Age"],
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str),
        FamilySize=family_size,
        FamilyTier=family_tier,
        HasCabin=X["Cabin"].notna().astype(int),
        TicketFreq=ticket["TicketFreq"],
        Embarked=X["Embarked"].fillna("Missing"),
        Fare=fare["Fare"],
    )[FEATURE_COLS]

    # 4. Model-specific representations, preserving feature order.
    categorical_strings = features[CAT_COLS].astype(str)
    numeric = features[NUM_COLS]
    cb_features = categorical_strings.skb.concat([numeric], axis=1)
    xgb_features = features[CAT_COLS].astype("category").skb.concat(
        [numeric], axis=1
    )

    # Unknown codes must not collide with known OrdinalEncoder codes. Replace
    # its -1 sentinel afterward to reproduce mapping.get(value, 0).
    nn_categories = categorical_strings.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    ).replace(-1, 0)
    nn_numeric = numeric.astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_categories.skb.concat([nn_numeric], axis=1)

    # This fit-only statistic is a recorded node passed to the predictor.
    cat_dims = nn_categories.max(axis=0) + 1

    # 5. The original reused scored validation labels for CatBoost stopping,
    #    the NN scheduler, and blend optimization. Preserve those mechanisms
    #    using an internal stratified 20% hold-out from outer-training rows.
    #    The original leaky validation score cannot be reproduced honestly.
    #    One GetXY per branch: same rows, y, seed and stratification, so all
    #    three carve the identical split.
    def carve(branch):
        parts = branch.skb.apply(GetXY(), y=y, no_wrap=True)
        return parts["X"], parts["X_val"], parts.get("y", y), parts["y_val"]

    cb_X, cb_X_val, fit_y, val_y = carve(cb_features)
    xgb_X, xgb_X_val, _, _ = carve(xgb_features)
    nn_X, nn_X_val, _, _ = carve(nn_features)

    # 6. Each member is its own node, fitted on the carved training rows and
    #    applied to the holdout, so it emits the holdout probabilities the
    #    blend search needs (and scored-row probabilities in predict mode).
    #    CatBoost keeps the original 50-round patience on the carved holdout.
    cb_probs = cb_X_val.skb.apply(
        FitOn(
            CatBoostClassifier(
                iterations=600,
                learning_rate=0.03,
                depth=5,
                eval_metric="Accuracy",
                random_seed=42,
                verbose=0,
            )
        ),
        y=val_y,
        fit_kwargs={
            "X_fit": cb_X,
            "y_fit": fit_y,
            "cat_features": CAT_COLS,
            "eval_set": (cb_X_val, val_y),
            "early_stopping_rounds": 50,
        },
    )
    xgb_probs = xgb_X_val.skb.apply(
        FitOn(
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                enable_categorical=True,
                tree_method="hist",
                random_state=42,
                eval_metric="logloss",
            )
        ),
        y=val_y,
        fit_kwargs={"X_fit": xgb_X, "y_fit": fit_y},
    )
    nn_probs = nn_X_val.skb.apply(
        FitOn(TorchTabularClassifier(random_state=42)),
        y=val_y,
        fit_kwargs={
            "X_fit": nn_X,
            "y_fit": fit_y,
            "X_val": nn_X_val,
            "y_val": val_y,
            "cat_dims": cat_dims,
        },
    )

    # 7. The original weight/threshold search, learned on the holdout.
    member_probs = skrub.deferred(stack)(cb_probs, xgb_probs, nn_probs)
    pred = member_probs.skb.apply(WeightThresholdSearch(), y=val_y)

    # 8. Score only the untouched outer hold-out; CV lives on mark_as_X.
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
