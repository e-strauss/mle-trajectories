import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class TitleAgeMedian(TransformerMixin, BaseEstimator):
    """Learn only the title-specific age medians."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby("Title")["Age"].median()
        return self

    def transform(self, X):
        return X.assign(Age=X["Age"].fillna(X["Title"].map(self.medians_)))


class GetXY(TransformerMixin, BaseEstimator):
    """Create a validation set using only the current fold's training rows."""

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


class NeuralProbability(ClassifierMixin, BaseEstimator):
    """The original torch training loop, refitted independently in each fold."""

    def __init__(self, batch_size=32, num_epochs=150, random_state=42):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]
        self.model_ = TabularNeuralNet(
            cat_dims, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        train_cat = torch.tensor(
            X[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        train_num = torch.tensor(
            X[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        val_cat = torch.tensor(
            X_val[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        val_num = torch.tensor(
            X_val[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )

        for _ in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for start in range(0, train_cat.size(0), self.batch_size):
                batch = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(train_cat[batch], train_num[batch])
                loss = criterion(outputs, train_y[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(val_cat, val_num)
                predictions = outputs.argmax(dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_val, predictions))
        return self

    def predict(self, X):
        self.model_.eval()
        cat = torch.tensor(
            X[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        num = torch.tensor(
            X[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        with torch.no_grad():
            outputs = self.model_(cat, num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def positive_probability(values):
    """Column 1 of a probability matrix, or a vector that is already that column."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        return values[:, 1]
    return values


def ensemble_predictions(cb_probs, xgb_probs, nn_probs, mode):
    if mode == "fit":
        return None
    blended = (
        positive_probability(cb_probs)
        + positive_probability(xgb_probs)
        + positive_probability(nn_probs)
    ) / 3.0
    return (blended >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this module records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and raw features. StratifiedShuffleSplit reproduces
    #    the original stratified train_test_split's exact hold-out rows.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Record feature engineering. The original full-table median imputation
    #    leaked validation information; each learned median now uses training
    #    rows only. The global age median is learned AFTER title-based filling.
    features = X.assign(
        Title=X["Name"].str.extract(
            r" ([A-Za-z]+)\.", expand=False
        ).fillna("Missing")
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    features = features.assign(
        CabinDeck=features["Cabin"].str[0].fillna("Missing")
    )
    features = features.assign(Embarked=features["Embarked"].fillna("Missing"))
    features = features.skb.apply(
        TitleAgeMedian(), cols=["Title", "Age"]
    )
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Age"]
    )
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Fare"]
    )

    categories = features[CAT_COLS].astype(str)
    encoded = categories.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    ).replace(-1, 0)
    # Sorted string categories match LabelEncoder; unseen categories map to 0.
    cat_dims = encoded.nunique().to_list()

    numbers = features[NUM_COLS]
    scaled_numbers = numbers.astype(np.float32).skb.apply(StandardScaler())

    cb_features = categories.skb.concat([numbers], axis=1)[FEATURE_COLS]
    xgb_features = encoded.skb.concat([numbers], axis=1)[FEATURE_COLS]
    nn_features = encoded.astype(np.int64).skb.concat(
        [scaled_numbers], axis=1
    )[FEATURE_COLS]

    # The original CatBoost early stopping and NN scheduler inspected the scored
    # hold-out. Keep their regularization, but expose an internal 20% stratified
    # validation split from training rows instead. Their original leaky score
    # therefore cannot be reproduced exactly.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    # cat_features is a fit argument so sklearn clone does not see it.
    # prediction_type="Probability" returns both classes; the blend takes column 1.
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

    xgb_probs = xgb_features.skb.apply(
        xgb.XGBModel(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        ),
        y=y,
    )

    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_probs = nn_parts["X"].skb.apply(
        NeuralProbability(batch_size=32, num_epochs=150, random_state=42),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "X_val": nn_parts["X_val"],
            "y_val": nn_parts["y_val"],
            "cat_dims": cat_dims,
        },
    )

    # Keep the exact equal-weight probability average and >= 0.5 threshold.
    # Gate arithmetic because predictor nodes contain estimators in fit mode.
    pred = cb_probs.skb.apply_func(
        ensemble_predictions, xgb_probs, nn_probs, skrub.eval_mode()
    )

    # 4. Score the single ensemble, not its constituent models independently.
    #    The splitter belongs exclusively to mark_as_X.
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
