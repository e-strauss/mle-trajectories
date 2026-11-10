import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS
PROB_COLS = ["CB", "XGB", "RF", "NN"]
XGB_DTYPES = {col: "category" for col in CAT_COLS}
N_SPLITS = 5


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
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


class CatBoostClassifierCloneable(CatBoostClassifier):
    def __sklearn_clone__(self):
        return CatBoostClassifierCloneable(**self.get_params(deep=False))


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, d) for n, d in zip(cat_dims, emb_dims)]
        )
        layers = []
        previous = sum(emb_dims) + num_features
        for width in hidden_dims:
            layers.extend([
                nn.Linear(previous, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
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


class ProbabilityCalibrator(TransformerMixin, BaseEstimator):
    """The original per-model LogisticRegression calibration."""

    def fit(self, X, y):
        self.calibrator_ = LogisticRegression(C=1.0, random_state=42).fit(X, y)
        self.columns_ = X.columns
        return self

    def transform(self, X):
        return pd.DataFrame(
            self.calibrator_.predict_proba(X)[:, 1],
            columns=self.columns_, index=X.index,
        )


class ThresholdSearch(ClassifierMixin, BaseEstimator):
    """The original threshold search on the meta-learner's OOF probabilities."""

    def fit(self, X, y):
        probabilities = X["meta_prob"].to_numpy()
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        self.threshold_ = 0.5
        best_score = -1.0
        for threshold in np.linspace(0.1, 0.9, 161):
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
    # 1. Load Data -- recorded read; import performs no reading or fitting.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark raw labels; reproduce the single outer hold-out.
    #    The five stacking folds below are internal training, not outer scoring.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Feature engineering. These independent learned steps fix the original
    #    full-table age/fare imputation and ticket-frequency leakage.
    age = X[["Pclass", "Sex", "Age"]].skb.apply(
        GroupedAgeImputer(), no_wrap=True
    )
    ticket_frequency = X[["Ticket"]].skb.apply(
        TicketFrequency(), no_wrap=True
    )
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))

    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = (
        raw_title.map({
            "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        })
        .fillna("Rare")
        .where(raw_title.notna(), "Missing")
    )
    family_size = X["SibSp"] + X["Parch"] + 1
    family_tier = (
        family_size.map({1: "IsAlone"})
        .where(family_size == 1, "SmallFamily")
        .where(family_size <= 4, "LargeFamily")
    )
    features = X.assign(
        Title=title,
        Age=age["Age"],
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str),
        FamilySize=family_size,
        FamilyTier=family_tier,
        IsAlone=(family_size == 1).astype(int),
        HasCabin=X["Cabin"].notna().astype(int),
        TicketFreq=ticket_frequency["TicketFreq"],
        Embarked=X["Embarked"].fillna("Missing"),
        Fare=fare["Fare"],
    )
    base_features = features[FEATURE_COLS]

    # 4. Model-specific representations, each in the original's column order.
    string_categories = base_features[CAT_COLS].astype(str)
    numeric_features = base_features[NUM_COLS]
    cb_features = string_categories.skb.concat([numeric_features], axis=1)
    nn_numeric = numeric_features.astype(np.float32)

    # 5. The original's internal 5-fold stacking loop, unrolled: per fold one
    #    split node per representation and one node per model. CatBoost
    #    early-stops on the fold's OOF rows, as in the original; those are
    #    outer-training rows, never the scored hold-out. Every branch splits
    #    with the same InnerFold, so all select the same rows (the split
    #    depends only on the row count and y).
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
                        eval_metric="Accuracy",
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
                },
            )
        )

        # The original casts each frame to "category" on its own, so every
        # split carries the vocabulary of its own rows.
        xgb_parts = base_features.skb.apply(InnerFold(fold), y=y, no_wrap=True)
        xgb_folds.append(
            xgb_parts["X_val"].astype(XGB_DTYPES).skb.apply(
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
                    "X_fit": xgb_parts["X"].astype(XGB_DTYPES),
                    "y_fit": fit_y,
                },
            )
        )

        # The fold's one-hot vocabulary is fitted on its training rows only.
        rf_cat = string_categories.skb.apply(
            InnerFold(fold), y=y, no_wrap=True
        ).skb.apply(
            SplitEncoder(
                OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ),
            no_wrap=True,
        )
        rf_num = numeric_features.skb.apply(InnerFold(fold), y=y, no_wrap=True)
        rf_folds.append(
            rf_cat["X_val"].skb.concat([rf_num["X_val"]], axis=1).skb.apply(
                FitOn(
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        random_state=42 + fold,
                    )
                ),
                y=val_y,
                fit_kwargs={
                    "X_fit": rf_cat["X"].skb.concat([rf_num["X"]], axis=1),
                    "y_fit": fit_y,
                },
            )
        )

        # The fold's LabelEncoder and StandardScaler are fitted on its
        # training rows only. OrdinalEncoder reserves -1 for unseen
        # categories, which then become 0 like mapping.get(v, 0).
        nn_cat = string_categories.skb.apply(
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
        nn_fit = nn_cat["X"].skb.concat([nn_num["X"]], axis=1)[FEATURE_COLS]
        nn_val = (
            nn_cat["X_val"]
            .replace(-1, 0)
            .skb.concat([nn_num["X_val"]], axis=1)[FEATURE_COLS]
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

    # 6. While fitting, every outer-training row is covered by exactly one
    #    fold (its OOF probability); while predicting, every scored row by all
    #    five. The original adds p / 5 per fold into a float64 array, with the
    #    division done in the model's own dtype (float32 for XGBoost and the
    #    network), so each fold is divided by its coverage in that dtype and
    #    summed in fold order. Row labels are attached explicitly: stratum
    #    returns predictions as bare arrays.
    coverage = skrub.deferred(pd.Series)(1, index=val_index[0]).reindex(
        features.index, fill_value=0
    )
    for index in val_index[1:]:
        coverage = coverage + skrub.deferred(pd.Series)(1, index=index).reindex(
            features.index, fill_value=0
        )

    def fold_average(fold_probs, name):
        total = None
        for probs, index in zip(fold_probs, val_index):
            probs = skrub.deferred(pd.Series)(probs, index=index).reindex(
                features.index, fill_value=0.0
            )
            share = (probs / coverage.astype(probs.dtype)).astype("float64")
            total = share if total is None else total + share
        return total.to_frame(name)

    # 7. Independent probability calibrators fitted to the four OOF columns.
    calibrated = [
        fold_average(folds, name).skb.apply(
            ProbabilityCalibrator(), y=y, no_wrap=True
        )
        for folds, name in zip(
            (cb_folds, xgb_folds, rf_folds, nn_folds), PROB_COLS
        )
    ]
    calibrated = calibrated[0].skb.concat(calibrated[1:], axis=1)[PROB_COLS]

    # 8. Recorded meta-features; NumPy uses population variance/std (ddof=0).
    meta_probabilities = calibrated.assign(
        prob_mean=calibrated.mean(axis=1),
        prob_variance=calibrated.var(axis=1, ddof=0),
        prob_std=calibrated.std(axis=1, ddof=0),
    )

    # The original runs get_dummies separately on the hold-out, so its dummy
    # schema came from the scored rows. Learn it on the outer training rows;
    # drop_first and get_dummies' numeric-first column order are kept.
    anchor_numbers = features[["Pclass", "IsAlone"]]
    anchor_categories = features[["Sex", "Title"]].skb.apply(
        OneHotEncoder(
            drop="first", handle_unknown="ignore", sparse_output=False
        )
    )
    meta_features = meta_probabilities.skb.concat(
        [anchor_numbers, anchor_categories], axis=1
    )

    # 9. The original meta-learner, then the threshold calibrated on its
    #    in-sample probabilities for the training rows.
    meta_probs = meta_features.skb.apply(
        FitOn(
            LogisticRegression(
                C=0.5, penalty="l2", random_state=42, max_iter=1000
            )
        ),
        y=y,
    )
    meta_frame = skrub.deferred(pd.Series)(
        meta_probs, index=features.index
    ).to_frame("meta_prob")
    pred = meta_frame.skb.apply(ThresholdSearch(), y=y)

    # 10. Score the original outer hold-out; its splitter lives only on X.
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
