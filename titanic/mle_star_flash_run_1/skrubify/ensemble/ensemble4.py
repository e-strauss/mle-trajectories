import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


CAT_COLS = [
    "Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier", "Deck"
]
NUM_COLS = [
    "Age", "SibSp", "Parch", "Fare", "FarePerPerson",
    "FamilySize", "HasCabin", "TicketFreq",
]
RF_CAT_COLS = [
    "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier", "Deck"
]
NN_COLS = CAT_COLS + NUM_COLS
META_COLS = ["p_cb", "p_xgb", "p_rf", "p_nn"]
N_SPLITS = 5


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        grouped_fill = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        self.fallback_ = X["Age"].fillna(grouped_fill).median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        grouped_fill = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X.assign(
            Age=X["Age"].fillna(grouped_fill).fillna(self.fallback_)
        )


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X.assign(TicketFreq=X["Ticket"].map(self.counts_).fillna(1))


class XGBCategories(TransformerMixin, BaseEstimator):
    """The original's sorted per-column categorical dtypes."""

    def fit(self, X, y=None):
        self.dtypes_ = {
            col: pd.CategoricalDtype(categories=sorted(X[col].unique()))
            for col in X.columns
        }
        return self

    def transform(self, X):
        return pd.DataFrame(
            {col: X[col].astype(dtype) for col, dtype in self.dtypes_.items()},
            index=X.index,
        )


class InnerFold(TransformerMixin, BaseEstimator):
    """One fold of the original's internal bagging ``StratifiedKFold``.

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


class CatBoostClassifierCloneable(CatBoostClassifier):
    def __sklearn_clone__(self):
        return CatBoostClassifierCloneable(**self.get_params(deep=False))


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, embedding_dim)
                for num_categories, embedding_dim in zip(cat_dims, emb_dims)
            ]
        )
        layers = []
        previous_dim = sum(emb_dims) + num_features
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


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


class SimplexWeights(RegressorMixin, BaseEstimator):
    """The original SLSQP Brier-score weights on the probability simplex."""

    def fit(self, X, y):
        probabilities = X[META_COLS].to_numpy()
        target = np.asarray(y)

        def brier_objective(weights):
            blended = np.dot(probabilities, np.array(weights))
            return np.mean((blended - target) ** 2)

        n_models = probabilities.shape[1]
        result = minimize(
            brier_objective,
            np.ones(n_models) / n_models,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        self.weights_ = result.x / np.sum(result.x)
        return self

    def predict(self, X):
        return np.dot(X[META_COLS].to_numpy(), self.weights_)


class ThresholdSearch(ClassifierMixin, BaseEstimator):
    """The original threshold search on the blend's OOF probabilities."""

    def fit(self, X, y):
        probabilities = X["blend_prob"].to_numpy()
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        self.threshold_ = 0.5
        best_score = -1.0
        for threshold in np.linspace(0.35, 0.65, 301):
            score = accuracy_score(
                labels, (probabilities >= threshold).astype(int)
            )
            if score > best_score:
                best_score = score
                self.threshold_ = threshold
        return self

    def predict(self, X):
        return (X["blend_prob"].to_numpy() >= self.threshold_).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- importing records the read without executing it.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the RAW target. The reported validation was
    #    one stratified 80/20 hold-out, not the internal five-fold bagging.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    title = raw_title.map(title_mapping).fillna("Rare")
    title = title.where(raw_title.notna(), "Missing")
    features = X.assign(Title=title)

    # These statistics originally included hold-out rows. Separate fitted
    # operators now learn them from the outer training rows only.
    features = features.skb.apply(GroupAgeImputer())
    features = features.assign(
        Deck=features["Cabin"].astype(str).str[0].where(
            features["Cabin"].notna()
            & features["Cabin"].astype(str).str.len().gt(0),
            "Missing",
        )
    )
    features = features.assign(
        HasCabin=features["Cabin"].notna().astype(int)
    )
    features = features.assign(
        Sex_Pclass=features["Sex"].astype(str)
        + "_"
        + features["Pclass"].astype(str)
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    features = features.assign(
        FamilyTier=features["FamilySize"].map(
            {1: "IsAlone"}
        ).fillna("SmallFamily").where(
            features["FamilySize"].le(4), "LargeFamily"
        )
    )
    features = features.skb.apply(TicketFrequency())
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Fare"]
    )
    features = features.assign(
        FarePerPerson=features["Fare"] / features["FamilySize"]
    )
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing")
    )

    # 4. Model-specific representations, each in the original's column order.
    string_features = features[CAT_COLS].astype(str)
    numeric = features[NUM_COLS]
    cb_features = string_features.skb.concat([numeric], axis=1)

    # XGBoost's categorical vocabularies, learned without the outer hold-out.
    xgb_features = string_features.skb.apply(XGBCategories()).skb.concat(
        [numeric], axis=1
    )

    # The original's get_dummies vocabulary spanned every row, hold-out
    # included. Learn it on the outer training rows; drop_first, the numeric
    # Pclass passthrough and the numeric-first column order are kept.
    rf_dummies = string_features[RF_CAT_COLS].skb.apply(
        OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    )
    rf_features = numeric.assign(Pclass=features["Pclass"]).skb.concat(
        [rf_dummies], axis=1
    )

    nn_numeric = numeric.astype(np.float32)

    # 5. The original's internal 5-fold bagging loop, unrolled: per fold one
    #    split node per representation and one node per model. CatBoost and
    #    XGBoost early-stop on the fold's OOF rows, as in the original; those
    #    are outer-training rows, never the scored hold-out. Every branch
    #    splits with the same InnerFold, so all select the same rows (the
    #    split depends only on the row count and y).
    cb_folds, xgb_folds, rf_folds, nn_folds, val_index = [], [], [], [], []
    for fold in range(N_SPLITS):
        cb_parts = cb_features.skb.apply(InnerFold(fold), y=y, no_wrap=True)
        fit_y, val_y = cb_parts["y"], cb_parts["y_val"]
        val_index.append(cb_parts["X_val"].index)

        cb_folds.append(
            cb_parts["X_val"].skb.apply(
                FitOn(
                    CatBoostClassifierCloneable(
                        iterations=600,
                        learning_rate=0.03,
                        depth=5,
                        cat_features=CAT_COLS,
                        eval_metric="Logloss",
                        random_seed=42 + fold,
                        verbose=0,
                    )
                ),
                y=val_y,
                fit_kwargs={
                    "X_fit": cb_parts["X"],
                    "y_fit": fit_y,
                    "eval_set": (cb_parts["X_val"], val_y),
                    "early_stopping_rounds": 50,
                    "verbose": False,
                },
            )
        )

        xgb_parts = xgb_features.skb.apply(InnerFold(fold), y=y, no_wrap=True)
        xgb_folds.append(
            xgb_parts["X_val"].skb.apply(
                FitOn(
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.03,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        enable_categorical=True,
                        tree_method="hist",
                        random_state=42 + fold,
                        eval_metric="logloss",
                        early_stopping_rounds=40,
                    )
                ),
                y=val_y,
                fit_kwargs={
                    "X_fit": xgb_parts["X"],
                    "y_fit": fit_y,
                    "eval_set": [(xgb_parts["X_val"], val_y)],
                    "verbose": False,
                },
            )
        )

        rf_parts = rf_features.skb.apply(InnerFold(fold), y=y, no_wrap=True)
        rf_folds.append(
            rf_parts["X_val"].skb.apply(
                FitOn(
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_split=4,
                        random_state=42 + fold,
                        n_jobs=-1,
                    )
                ),
                y=val_y,
                fit_kwargs={"X_fit": rf_parts["X"], "y_fit": fit_y},
            )
        )

        # The fold's LabelEncoder and StandardScaler are fitted on its
        # training rows only, as in the original. OrdinalEncoder reserves -1
        # for unseen categories, which then become 0 like mapping.get(c, 0).
        nn_cat = string_features.skb.apply(
            InnerFold(fold), y=y, no_wrap=True
        ).skb.apply(
            SplitEncoder(
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.int64,
                )
            ),
            no_wrap=True,
        )
        nn_num = nn_numeric.skb.apply(
            InnerFold(fold), y=y, no_wrap=True
        ).skb.apply(SplitEncoder(StandardScaler()), no_wrap=True)
        nn_fit = nn_cat["X"].skb.concat([nn_num["X"]], axis=1)[NN_COLS]
        nn_val = (
            nn_cat["X_val"]
            .replace(-1, 0)
            .skb.concat([nn_num["X_val"]], axis=1)[NN_COLS]
        )
        cat_dims = nn_cat["X"].max(axis=0) + 1
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
    #    and the mean is the original's fold average. The fold's row labels
    #    are attached explicitly: stratum returns predictions as bare arrays.
    #    The original stores every probability in a float64 array.
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
            p_rf=out_of_fold(rf_folds),
            p_nn=out_of_fold(nn_folds),
        )
    )[META_COLS]

    # 7. The original simplex weights, then the decision threshold calibrated
    #    on the training rows' OOF blend.
    blend = meta.skb.apply(SimplexWeights(), y=y)
    blend_frame = skrub.deferred(pd.Series)(
        blend, index=features.index
    ).to_frame("blend_prob")
    pred = blend_frame.skb.apply(ThresholdSearch(), y=y)

    # 8. Score -- mark_as_X owns the single outer stratified hold-out.
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
