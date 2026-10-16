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
from sklearn.preprocessing import LabelEncoder, StandardScaler


class GetXY(TransformerMixin, BaseEstimator):
    """Split this fold's training rows, returning a pandas container."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        X_fit, X_val, y_fit, y_val = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        # Preserve train_test_split's row order within both parts. Metadata is
        # removed by recorded selections before either predictor receives X.
        return pd.concat(
            [
                X_fit.assign(
                    __evaluation_row=False,
                    __fit_target=np.asarray(y_fit),
                ),
                X_val.assign(
                    __evaluation_row=True,
                    __fit_target=np.asarray(y_val),
                ),
            ],
            axis=0,
        )

    def transform(self, X):
        # At prediction time all supplied rows go to the predictor.
        return X.assign(__evaluation_row=False, __fit_target=np.nan)


class CatBoostPositiveProbability(CatBoostClassifier):
    """Expose the probability column used by the original script."""

    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))

    def predict(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)[:, 1]


class TrainingLabelEncoder(TransformerMixin, BaseEstimator):
    """Learn categorical vocabularies; unseen values map to position zero."""

    def fit(self, X, y=None):
        self.encoders_ = {
            col: LabelEncoder().fit(X[col]) for col in X.columns
        }
        return self

    def transform(self, X):
        return pd.DataFrame(
            {
                col: X[col]
                .map(
                    {
                        value: position
                        for position, value in enumerate(encoder.classes_)
                    }
                )
                .fillna(0)
                .astype(np.int64)
                for col, encoder in self.encoders_.items()
            },
            index=X.index,
        )


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
        embs = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embs + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """The original torch loop, with preprocessing and splitting outside."""

    def __init__(self, cat_cols, num_cols, random_state=42):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Sizes are supplied by a recorded statistic on the outer fold's
        # training data, not computed from the scored hold-out.
        dims = np.asarray(cat_dims, dtype=np.int64).tolist()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in dims]
        self.model_ = TabularNeuralNet(
            dims, emb_dims, num_features=len(self.num_cols)
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        train_cat = torch.tensor(
            X[self.cat_cols].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        train_num = torch.tensor(
            X[self.num_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        val_cat = torch.tensor(
            X_val[self.cat_cols].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        val_num = torch.tensor(
            X_val[self.num_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )

        for epoch in range(150):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for i in range(0, train_cat.size(0), 32):
                indices_batch = permutation[i : i + 32]
                optimizer.zero_grad()
                outputs = self.model_(
                    train_cat[indices_batch], train_num[indices_batch]
                )
                loss = criterion(outputs, train_y[indices_batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(val_cat, val_num)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_val, val_preds))

        self.model_.eval()
        return self

    def predict(self, X):
        x_cat = torch.tensor(
            X[self.cat_cols].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        x_num = torch.tensor(
            X[self.num_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(x_cat, x_num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def threshold_probability(probability, mode):
    if mode == "fit":
        return None
    return (probability >= 0.5).astype(int)


def threshold_ensemble(cb_probability, nn_probability, mode):
    if mode == "fit":
        return None
    return (
        (0.5 * cb_probability + 0.5 * nn_probability) >= 0.5
    ).astype(int)


def build_probability_nodes(features, cat_cols, num_cols, y):
    """Build visible graph nodes; this helper is not an executed DataOps UDF."""
    feature_cols = cat_cols + num_cols
    categorical = features[cat_cols].astype(str)
    numeric = features[num_cols]

    cb_features = categorical.skb.concat([numeric], axis=1)

    # This skrub version does not accept how="no_wrap". GetXY therefore returns
    # a DataFrame, and ordinary recorded selections extract the four pieces.
    cb_parts = cb_features.skb.apply(GetXY(), y=y)
    cb_fit = cb_parts[~cb_parts["__evaluation_row"]]
    cb_val = cb_parts[cb_parts["__evaluation_row"]]
    cb_probability = cb_fit[feature_cols].skb.apply(
        CatBoostPositiveProbability(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=cat_cols,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_fit["__fit_target"],
        fit_kwargs={
            "eval_set": (
                cb_val[feature_cols],
                cb_val["__fit_target"],
            ),
            "early_stopping_rounds": 50,
        },
    )

    encoded = categorical.skb.apply(TrainingLabelEncoder())
    scaled = numeric.astype(np.float32).skb.apply(StandardScaler())
    nn_features = encoded.skb.concat([scaled], axis=1)
    cat_dims = encoded.max(axis=0) + 1

    nn_parts = nn_features.skb.apply(GetXY(), y=y)
    nn_fit = nn_parts[~nn_parts["__evaluation_row"]]
    nn_val = nn_parts[nn_parts["__evaluation_row"]]
    nn_probability = nn_fit[feature_cols].skb.apply(
        NeuralPositiveProbability(cat_cols=cat_cols, num_cols=num_cols),
        y=nn_fit["__fit_target"],
        fit_kwargs={
            "X_val": nn_val[feature_cols],
            "y_val": nn_val["__fit_target"],
            "cat_dims": cat_dims,
        },
    )
    return cb_probability, nn_probability


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records the read without executing it.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — mark the raw target. This splitter reproduces the
    #    original single train_test_split(..., stratify=y).
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    #    Replace the original leaky full-table medians with fold-fitted medians.
    imputed = X[["Age", "Fare"]].skb.apply(
        SimpleImputer(strategy="median")
    )
    prepared = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    prepared = prepared.assign(Age=imputed["Age"])
    prepared = prepared.assign(Fare=imputed["Fare"])

    engineered = prepared.assign(
        Title=prepared["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
        .fillna("Missing")
    )
    engineered = engineered.assign(
        FamilySize=engineered["SibSp"] + engineered["Parch"] + 1
    )
    engineered = engineered.assign(
        CabinDeck=engineered["Cabin"].str.get(0).fillna("Missing")
    )

    # The original used scored rows for CatBoost early stopping and the neural
    # scheduler. Keep their parameters, but use internal evaluation rows from
    # the outer training fold. Its original leaky scores cannot be reproduced.
    cb_probs, nn_probs = build_probability_nodes(
        engineered,
        ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"],
        ["Age", "SibSp", "Parch", "Fare", "FamilySize"],
        y,
    )
    cb_raw_probs, nn_raw_probs = build_probability_nodes(
        prepared,
        ["Pclass", "Sex", "Embarked"],
        ["Age", "SibSp", "Parch", "Fare"],
        y,
    )

    # Enumerate exactly the original four variants, preserving probability
    # averaging and >= 0.5 thresholds. Post-prediction arithmetic is gated.
    variants = {
        "Baseline (Full Ensemble + FE)": cb_probs.skb.apply_func(
            threshold_ensemble, nn_probs, skrub.eval_mode()
        ),
        "Ablation 1 (CatBoost Only - No NN)": cb_probs.skb.apply_func(
            threshold_probability, skrub.eval_mode()
        ),
        "Ablation 2 (NN Only - No CatBoost)": nn_probs.skb.apply_func(
            threshold_probability, skrub.eval_mode()
        ),
        "Ablation 3 (No Feature Engineering)": cb_raw_probs.skb.apply_func(
            threshold_ensemble, nn_raw_probs, skrub.eval_mode()
        ),
    }
    pred = skrub.choose_from(variants, name="variant").as_data_op()

    # 4. Score all variants in one grid search, using the marked splitter.
    #    Non-scoring comparison summaries and progress printing are omitted.
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
