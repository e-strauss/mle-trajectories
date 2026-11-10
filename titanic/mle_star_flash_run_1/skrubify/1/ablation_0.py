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


class GetXY(TransformerMixin, BaseEstimator):
    """Reserve an early-stopping set from the current fold's training rows."""

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


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Only the torch training loop; encoding and scaling are separate nodes."""

    def __init__(
        self,
        cat_cols,
        num_cols,
        random_state=42,
        batch_size=32,
        num_epochs=150,
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def fit(self, X, y, eval_set, cat_dims):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)

        # X is GetXY's training rows. The scheduler hold-out is the sibling
        # node, and the embedding sizes were measured on the whole outer fold.
        categories = X[self.cat_cols].to_numpy(dtype=np.int64)
        numeric = X[self.num_cols].to_numpy(dtype=np.float32)
        X_val, y_val = eval_set
        val_categories = X_val[self.cat_cols].to_numpy(dtype=np.int64)
        val_numeric = X_val[self.num_cols].to_numpy(dtype=np.float32)
        dims = np.asarray(cat_dims).ravel().astype(int).tolist()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in dims]
        labels = np.asarray(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_ = TabularNeuralNet(
            dims, emb_dims, num_features=numeric.shape[1]
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        train_cat = torch.tensor(
            categories, dtype=torch.long, device=self.device_
        )
        train_num = torch.tensor(
            numeric, dtype=torch.float32, device=self.device_
        )
        train_y = torch.tensor(
            labels, dtype=torch.long, device=self.device_
        )
        val_cat = torch.tensor(
            val_categories, dtype=torch.long, device=self.device_
        )
        val_num = torch.tensor(
            val_numeric, dtype=torch.float32, device=self.device_
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
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(np.asarray(y_val), predictions))

        return self

    def predict(self, X):
        self.model_.eval()
        categories = torch.tensor(
            X[self.cat_cols].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        numeric = torch.tensor(
            X[self.num_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        with torch.no_grad():
            outputs = self.model_(categories, numeric)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def threshold_probability(probability, mode):
    if mode == "fit":
        return None
    return (probability >= 0.5).astype(int)


def threshold_catboost(probability, mode):
    # prediction_type="Probability" returns (n_samples, n_classes) in classes_
    # order. classes_ is [0, 1], so column 1 is the positive-class probability.
    if mode == "fit":
        return None
    probability = np.asarray(probability, dtype=float)[:, 1]
    return (probability >= 0.5).astype(int)


def threshold_ensemble(cb_probability, nn_probability, mode):
    # prediction_type="Probability" returns (n_samples, n_classes) in classes_
    # order. classes_ is [0, 1], so column 1 is the positive-class probability
    # the net's predict already returns.
    if mode == "fit":
        return None
    cb_probability = np.asarray(cb_probability, dtype=float)[:, 1]
    return (
        (0.5 * cb_probability + 0.5 * nn_probability) >= 0.5
    ).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this file records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and features. StratifiedShuffleSplit reproduces
    #    the original stratified train_test_split's exact outer hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    #    The original full-table median imputation leaked validation information;
    #    these medians are instead fitted on each outer fold's training rows.
    features = X.skb.apply(
        SimpleImputer(strategy="median"), cols=["Age", "Fare"]
    )
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing")
    )
    engineered = features.assign(
        Title=features["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
        .fillna("Missing"),
        FamilySize=features["SibSp"] + features["Parch"] + 1,
        CabinDeck=features["Cabin"].str[0].fillna("Missing"),
    )

    fe_cat_cols = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"]
    fe_num_cols = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
    raw_cat_cols = ["Pclass", "Sex", "Embarked"]
    raw_num_cols = ["Age", "SibSp", "Parch", "Fare"]

    # Preserve the original categorical-then-numeric column order.
    fe_categorical = engineered[fe_cat_cols].astype(str)
    raw_categorical = features[raw_cat_cols].astype(str)
    fe_cb_features = fe_categorical.skb.concat(
        [engineered[fe_num_cols]], axis=1
    )
    raw_cb_features = raw_categorical.skb.concat(
        [features[raw_num_cols]], axis=1
    )

    # CatBoost originally early-stopped on the scored hold-out. Keep the
    # patience and iteration limit, but use fold-training-only eval rows.
    fe_cb_parts = fe_cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    raw_cb_parts = raw_cb_features.skb.apply(GetXY(), y=y, no_wrap=True)

    fe_cb_probability = fe_cb_parts["X"].skb.apply(
        CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=fe_cb_parts.get("y", y),
        fit_kwargs={
            "cat_features": fe_cat_cols,
            "eval_set": (fe_cb_parts["X_val"], fe_cb_parts["y_val"]),
            "early_stopping_rounds": 50,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    raw_cb_probability = raw_cb_parts["X"].skb.apply(
        CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=raw_cb_parts.get("y", y),
        fit_kwargs={
            "cat_features": raw_cat_cols,
            "eval_set": (raw_cb_parts["X_val"], raw_cb_parts["y_val"]),
            "early_stopping_rounds": 50,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )

    # OrdinalEncoder sorts each string vocabulary as LabelEncoder did.
    # Unknown categories become zero, matching mapping.get(value, 0).
    fe_nn_categorical = (
        fe_categorical.skb.apply(
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.int64,
            )
        )
        .replace(-1, 0)
        .astype(np.int64)
    )
    raw_nn_categorical = (
        raw_categorical.skb.apply(
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.int64,
            )
        )
        .replace(-1, 0)
        .astype(np.int64)
    )

    fe_nn_numeric = (
        engineered[fe_num_cols].astype(np.float32).skb.apply(StandardScaler())
    )
    raw_nn_numeric = (
        features[raw_num_cols].astype(np.float32).skb.apply(StandardScaler())
    )
    fe_nn_features = fe_nn_categorical.skb.concat([fe_nn_numeric], axis=1)
    raw_nn_features = raw_nn_categorical.skb.concat([raw_nn_numeric], axis=1)

    # Both nets stepped their scheduler on the scored hold-out. Each now
    # trains on a recorded GetXY slice of its own feature frame. Embedding
    # sizes stay those of the whole outer fold, measured before the split.
    fe_cat_dimensions = fe_nn_features[fe_cat_cols].max(axis=0) + 1
    raw_cat_dimensions = raw_nn_features[raw_cat_cols].max(axis=0) + 1
    fe_nn_parts = fe_nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    raw_nn_parts = raw_nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    fe_nn_probability = fe_nn_parts["X"].skb.apply(
        NeuralPositiveProbability(fe_cat_cols, fe_num_cols),
        y=fe_nn_parts.get("y", y),
        fit_kwargs={
            "eval_set": (fe_nn_parts["X_val"], fe_nn_parts["y_val"]),
            "cat_dims": fe_cat_dimensions,
        },
    )
    raw_nn_probability = raw_nn_parts["X"].skb.apply(
        NeuralPositiveProbability(raw_cat_cols, raw_num_cols),
        y=raw_nn_parts.get("y", y),
        fit_kwargs={
            "eval_set": (raw_nn_parts["X_val"], raw_nn_parts["y_val"]),
            "cat_dims": raw_cat_dimensions,
        },
    )

    # Exactly the four original ablations, sharing probability-producing nodes.
    # Prediction arithmetic is gated because fit-mode values are estimators.
    variants = {
        "Baseline (Full Ensemble + FE)": fe_cb_probability.skb.apply_func(
            threshold_ensemble, fe_nn_probability, skrub.eval_mode()
        ),
        "Ablation 1 (CatBoost Only - No NN)": fe_cb_probability.skb.apply_func(
            threshold_catboost, skrub.eval_mode()
        ),
        "Ablation 2 (NN Only - No CatBoost)": fe_nn_probability.skb.apply_func(
            threshold_probability, skrub.eval_mode()
        ),
        "Ablation 3 (No Feature Engineering)": raw_cb_probability.skb.apply_func(
            threshold_ensemble, raw_nn_probability, skrub.eval_mode()
        ),
    }
    pred = skrub.choose_from(variants, name="variant").as_data_op()

    # 4. Score every original variant; the splitter above owns validation.
    #    Component-ranking/progress output is omitted: results_ holds the scores.
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
