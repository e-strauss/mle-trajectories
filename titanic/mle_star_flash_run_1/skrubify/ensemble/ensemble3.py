import lightgbm as lgb
import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS
META_COLS = ["p_cb", "p_xgb", "p_lgbm", "p_nn"]
N_SPLITS = 5


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


class CategoricalVocabulary(TransformerMixin, BaseEstimator):
    """Preserve the training partition's native categorical vocabularies."""

    def fit(self, X, y=None):
        self.categories_ = {
            col: X[col].astype("category").cat.categories for col in X.columns
        }
        return self

    def transform(self, X):
        return pd.DataFrame(
            {
                col: pd.Categorical(X[col], categories=categories)
                for col, categories in self.categories_.items()
            },
            index=X.index,
        )


class InnerFold(TransformerMixin, BaseEstimator):
    """One fold of the original's internal stacking ``StratifiedKFold``.

    In predict mode ``X_val`` is the scored rows: each fold's models are
    applied to ``X_val``, so the same model node yields its out-of-fold
    probabilities while fitting and scored-row probabilities while predicting.
    """

    def __init__(self, fold, n_splits=N_SPLITS, random_state=42):
        self.fold = fold
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        train_idx, val_idx = list(splitter.split(X, y))[self.fold]
        return {
            "X": X.iloc[train_idx],
            "X_val": X.iloc[val_idx],
            "y": y.iloc[train_idx],
            "y_val": y.iloc[val_idx],
        }

    def transform(self, X):
        return {"X": X, "X_val": X, "y": None, "y_val": None}


class SplitEncoder(TransformerMixin, BaseEstimator):
    """Fit ``transformer`` on a fold's training part and transform both parts."""

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        self.transformer_ = clone(self.transformer)
        if hasattr(self.transformer_, "set_output"):
            self.transformer_.set_output(transform="pandas")
        self.transformer_.fit(X["X"])
        return {
            "X": self.transformer_.transform(X["X"]),
            "X_val": self.transformer_.transform(X["X_val"]),
        }

    def transform(self, X):
        encoded = self.transformer_.transform(X["X_val"])
        return {"X": encoded, "X_val": encoded}


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
    """The original torch training loop, without feature preprocessing.

    The original seeds torch once and its five folds then share one RNG
    stream, so only the first fold passes ``seed``. ``after`` is the previous
    fold's network node; it is unused, and only forces the folds to train in
    order so the stream continues exactly as in the original's loop.
    """

    def __init__(self, seed=None):
        self.seed = seed

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

    def fit(self, X, y, cat_dims=None, after=None):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dimensions = np.asarray(cat_dims, dtype=int).tolist()
        embedding_dimensions = [
            max(2, int(round(dim**0.5 * 2))) for dim in dimensions
        ]
        self.model_ = TabularNeuralNet(
            dimensions, embedding_dimensions, num_features=len(NUM_COLS)
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        train_cat, train_num = self._tensors(X)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )

        for epoch in range(120):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for start in range(0, train_cat.size(0), 32):
                batch = permutation[start : start + 32]
                optimizer.zero_grad()
                outputs = self.model_(train_cat[batch], train_num[batch])
                loss = criterion(outputs, train_y[batch])
                loss.backward()
                optimizer.step()
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
    """Fit ``estimator`` on ``X_fit``/``y_fit``; predict P(class 1) on X.

    A fold model's node input is that fold's validation rows (the scored rows
    in predict mode), so its training rows arrive through ``fit_kwargs``.
    Without them it fits on the node's own rows.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, X_fit=None, y_fit=None, **fit_params):
        if X_fit is None:
            X_fit, y_fit = X, y
        self.estimator_ = clone(self.estimator).fit(X_fit, y_fit, **fit_params)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict_proba(X)[:, 1]


class ThresholdSearch(ClassifierMixin, BaseEstimator):
    """The original threshold search on the meta-learner's OOF probabilities."""

    def fit(self, X, y):
        probabilities = X["meta_prob"].to_numpy()
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        self.threshold_ = 0.5
        best_score = -1.0
        for threshold in np.linspace(0.2, 0.8, 61):
            score = accuracy_score(
                labels, (probabilities >= threshold).astype(int)
            )
            if score > best_score:
                best_score = score
                self.threshold_ = threshold
        return self

    def predict(self, X):
        return (X["meta_prob"].to_numpy() >= self.threshold_).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- importing records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- raw target and the original stratified 80/20 holdout.
    #    The five stacking folds below are internal training, not outer scoring.
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
        (family_size <= 4)
        .map({True: "SmallFamily", False: "LargeFamily"})
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

    # 4. Model-specific representations, preserving feature order. The tree
    #    vocabularies, the NN label codes and cat_dims span the outer training
    #    partition, as in the original.
    categorical_strings = features[CAT_COLS].astype(str)
    numeric = features[NUM_COLS]
    cb_features = categorical_strings.skb.concat([numeric], axis=1)
    tree_features = (
        features[CAT_COLS]
        .skb.apply(CategoricalVocabulary())
        .skb.concat([numeric], axis=1)
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
    cat_dims = nn_categories.max(axis=0) + 1
    nn_numeric = numeric.astype(np.float32)

    # 5. The original's internal 5-fold stacking loop, unrolled: per fold one
    #    split node per representation and one node per model. CB/LGBM
    #    early-stop and XGB logs on the fold's OOF rows, as in the original;
    #    those are outer-training rows, never the scored holdout. Every branch
    #    splits with the same InnerFold, so all select the same rows (the split
    #    depends only on the row count and y).
    cb_folds, xgb_folds, lgbm_folds, nn_folds, val_index = [], [], [], [], []
    for fold in range(N_SPLITS):
        parts = features.skb.apply(InnerFold(fold), y=y, no_wrap=True)
        fit_y, val_y = parts["y"], parts["y_val"]
        val_index.append(parts["X_val"].index)

        cb_parts = cb_features.skb.apply(InnerFold(fold), y=y, no_wrap=True)
        cb_folds.append(
            cb_parts["X_val"].skb.apply(
                FitOn(
                    CatBoostClassifier(
                        iterations=500,
                        learning_rate=0.03,
                        depth=5,
                        eval_metric="Accuracy",
                        random_seed=42 + fold,
                        verbose=0,
                    )
                ),
                y=val_y,
                fit_kwargs={
                    "X_fit": cb_parts["X"],
                    "y_fit": fit_y,
                    "cat_features": CAT_COLS,
                    "eval_set": (cb_parts["X_val"], val_y),
                    "early_stopping_rounds": 50,
                },
            )
        )

        tree_parts = tree_features.skb.apply(InnerFold(fold), y=y, no_wrap=True)
        xgb_folds.append(
            tree_parts["X_val"].skb.apply(
                FitOn(
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        enable_categorical=True,
                        tree_method="hist",
                        random_state=42 + fold,
                        eval_metric="logloss",
                    )
                ),
                y=val_y,
                fit_kwargs={
                    "X_fit": tree_parts["X"],
                    "y_fit": fit_y,
                    "eval_set": [(tree_parts["X_val"], val_y)],
                    "verbose": False,
                },
            )
        )
        lgbm_folds.append(
            tree_parts["X_val"].skb.apply(
                FitOn(
                    lgb.LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.03,
                        num_leaves=15,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42 + fold,
                        verbose=-1,
                    )
                ),
                y=val_y,
                fit_kwargs={
                    "X_fit": tree_parts["X"],
                    "y_fit": fit_y,
                    "eval_set": [(tree_parts["X_val"], val_y)],
                    "callbacks": [lgb.early_stopping(50, verbose=False)],
                },
            )
        )

        # The fold's StandardScaler is fitted on its training rows only.
        nn_cat = nn_categories.skb.apply(InnerFold(fold), y=y, no_wrap=True)
        nn_num = nn_numeric.skb.apply(
            InnerFold(fold), y=y, no_wrap=True
        ).skb.apply(SplitEncoder(StandardScaler()), no_wrap=True)
        nn_fit = nn_cat["X"].skb.concat([nn_num["X"]], axis=1)[FEATURE_COLS]
        nn_val = nn_cat["X_val"].skb.concat([nn_num["X_val"]], axis=1)[
            FEATURE_COLS
        ]
        nn_kwargs = {"X_fit": nn_fit, "y_fit": fit_y, "cat_dims": cat_dims}
        if nn_folds:
            nn_kwargs["after"] = nn_folds[-1]
        nn_folds.append(
            nn_val.skb.apply(
                FitOn(TorchTabularClassifier(seed=42 if fold == 0 else None)),
                y=val_y,
                fit_kwargs=nn_kwargs,
            )
        )

    # 6. While fitting, every outer-training row is predicted by exactly one
    #    fold (out-of-fold); while predicting, every scored row by all five,
    #    and the mean is the original's fold average. The fold's row labels are
    #    attached explicitly: stratum returns predictions as bare arrays. The
    #    original stores every probability in a float64 array.
    def out_of_fold(fold_probs):
        fold_probs = [
            skrub.deferred(pd.Series)(probs, index=index)
            for probs, index in zip(fold_probs, val_index)
        ]
        stacked = skrub.deferred(pd.concat)(fold_probs).astype("float64")
        return stacked.groupby(level=0).mean().reindex(features.index)

    meta = (
        out_of_fold(cb_folds)
        .to_frame("p_cb")
        .assign(
            p_xgb=out_of_fold(xgb_folds),
            p_lgbm=out_of_fold(lgbm_folds),
            p_nn=out_of_fold(nn_folds),
        )
    )[META_COLS]

    # 7. The original meta-learner, then the threshold calibrated on the
    #    training rows' OOF meta-probabilities.
    meta_probs = meta.skb.apply(
        FitOn(LogisticRegression(C=1.0, random_state=42)), y=y
    )
    meta_frame = skrub.deferred(pd.Series)(
        meta_probs, index=features.index
    ).to_frame("meta_prob")
    pred = meta_frame.skb.apply(ThresholdSearch(), y=y)

    # 8. Score -- the stratified holdout on mark_as_X drives validation.
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
