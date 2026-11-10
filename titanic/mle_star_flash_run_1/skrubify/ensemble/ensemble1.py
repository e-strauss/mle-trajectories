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
META_COLS = [
    "p_cb",
    "p_xgb",
    "p_nn",
    "mean_prob",
    "std_prob",
    "diff_cb_xgb",
    "diff_cb_nn",
    "diff_xgb_nn",
    "prod_cb_xgb",
    "prod_cb_nn",
    "prod_xgb_nn",
]
N_SPLITS = 5
BLEND_WEIGHT = 0.65


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
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            if fold == self.fold:
                break
        return {
            "X": X.iloc[train_idx],
            "X_val": X.iloc[val_idx],
            "y": y.iloc[train_idx],
            "y_val": y.iloc[val_idx],
        }

    def transform(self, X):
        return {"X": X, "X_val": X, "y": None, "y_val": None}


class FoldCategories(TransformerMixin, BaseEstimator):
    """Learn XGBoost's category vocabularies from its training rows."""

    def fit(self, X, y=None):
        self.categories_ = {
            col: X[col].astype("category").cat.categories for col in CAT_COLS
        }
        return self

    def transform(self, X):
        return X.assign(
            **{
                col: pd.Categorical(X[col], categories=self.categories_[col])
                for col in CAT_COLS
            }
        )


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
    """The original torch training loop with validation-loss checkpointing.

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

    def fit(self, X, y, X_val=None, y_val=None, cat_dims=None, after=None):
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        train_cat, train_num = self._tensors(X)
        val_cat, val_num = self._tensors(X_val)
        labels = np.asarray(y_val)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        val_y = torch.tensor(labels, dtype=torch.long, device=self.device_)

        best_loss = float("inf")
        best_weights = None
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

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(val_cat, val_num)
                val_loss = criterion(outputs, val_y).item()
                val_accuracy = (
                    torch.argmax(outputs, dim=1).cpu().numpy() == labels
                ).mean()
            scheduler.step(val_accuracy)

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = {
                    key: value.cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }

        if best_weights is not None:
            self.model_.load_state_dict(
                {key: value.to(self.device_) for key, value in best_weights.items()}
            )
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
    """The original threshold search on the training rows' blended probabilities."""

    def fit(self, X, y):
        probabilities = X["ensemble_prob"].to_numpy()
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        self.threshold_ = 0.5
        best_score = -1.0
        for threshold in np.linspace(0.2, 0.8, 121):
            score = accuracy_score(
                labels, (probabilities >= threshold).astype(int)
            )
            if score > best_score:
                best_score = score
                self.threshold_ = threshold
        return self

    def predict(self, X):
        return (X["ensemble_prob"].to_numpy() >= self.threshold_).astype(int)


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

    # 4. The original's internal 5-fold stacking loop over the outer-training
    #    rows, unrolled: per fold one split node, its encoders, and one node
    #    per model. The boosters early-stop and the network checkpoints on the
    #    fold's validation rows, as in the original; those never include the
    #    scored hold-out. The row-wise casts happen once, before splitting;
    #    every branch splits with the same InnerFold, so all select the same
    #    rows (the split depends only on the row count and y).
    cb_features = features.astype(dict.fromkeys(CAT_COLS, str))
    nn_cat_features = features[CAT_COLS].astype(str)
    nn_num_features = features[NUM_COLS].astype(np.float32)

    cb_folds, xgb_folds, nn_folds, val_index = [], [], [], []
    for fold in range(N_SPLITS):
        parts = features.skb.apply(InnerFold(fold), y=y, no_wrap=True)
        fit_y, val_y = parts["y"], parts["y_val"]
        val_index.append(parts["X_val"].index)

        cb_parts = cb_features.skb.apply(InnerFold(fold), y=y, no_wrap=True)
        cb_fit, cb_val = cb_parts["X"], cb_parts["X_val"]
        cb_folds.append(
            cb_val.skb.apply(
                FitOn(
                    CatBoostClassifier(
                        iterations=600,
                        learning_rate=0.03,
                        depth=5,
                        eval_metric="Accuracy",
                        random_seed=42 + fold,
                        verbose=0,
                    )
                ),
                y=val_y,
                fit_kwargs={
                    "X_fit": cb_fit,
                    "y_fit": fit_y,
                    "cat_features": CAT_COLS,
                    "eval_set": (cb_val, val_y),
                    "early_stopping_rounds": 50,
                },
            )
        )

        xgb_parts = parts.skb.apply(SplitEncoder(FoldCategories()), no_wrap=True)
        xgb_folds.append(
            xgb_parts["X_val"].skb.apply(
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
                        early_stopping_rounds=50,
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

        # LabelEncoder plus mapping.get(value, 0) for unseen validation values.
        nn_cat = nn_cat_features.skb.apply(
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
        nn_num = nn_num_features.skb.apply(
            InnerFold(fold), y=y, no_wrap=True
        ).skb.apply(SplitEncoder(StandardScaler()), no_wrap=True)
        nn_fit = nn_cat["X"].skb.concat([nn_num["X"]], axis=1)
        nn_val = nn_cat["X_val"].replace(-1, 0).skb.concat(
            [nn_num["X_val"]], axis=1
        )
        nn_kwargs = {
            "X_fit": nn_fit,
            "y_fit": fit_y,
            "X_val": nn_val,
            "y_val": val_y,
            "cat_dims": nn_cat["X"].max(axis=0) + 1,
        }
        if nn_folds:
            nn_kwargs["after"] = nn_folds[-1]
        nn_folds.append(
            nn_val.skb.apply(
                FitOn(TorchTabularClassifier(seed=42 if fold == 0 else None)),
                y=val_y,
                fit_kwargs=nn_kwargs,
            )
        )

    # 5. While fitting, every outer-training row is predicted by exactly one
    #    fold (out-of-fold); while predicting, every scored row by all five,
    #    and the mean is the original's fold average. The fold's row labels are
    #    attached explicitly: stratum returns predictions as bare arrays.
    def out_of_fold(fold_probs):
        fold_probs = [
            skrub.deferred(pd.Series)(probs, index=index)
            for probs, index in zip(fold_probs, val_index)
        ]
        stacked = skrub.deferred(pd.concat)(fold_probs).astype("float64")
        return stacked.groupby(level=0).mean().reindex(features.index)

    base = (
        out_of_fold(cb_folds)
        .to_frame("p_cb")
        .assign(p_xgb=out_of_fold(xgb_folds), p_nn=out_of_fold(nn_folds))
    )

    # 6. Meta-features. The soft-voting average uses the unclipped
    #    probabilities; only the meta-features are built from clipped ones.
    base_average = (base["p_cb"] + base["p_xgb"] + base["p_nn"]) / 3.0
    meta = base.clip(lower=1e-6, upper=1.0 - 1e-6)
    mean_probability = (meta["p_cb"] + meta["p_xgb"] + meta["p_nn"]) / 3.0
    meta = meta.assign(
        mean_prob=mean_probability,
        std_prob=(
            (
                (meta["p_cb"] - mean_probability) ** 2
                + (meta["p_xgb"] - mean_probability) ** 2
                + (meta["p_nn"] - mean_probability) ** 2
            )
            / 3.0
        ).skb.apply_func(np.sqrt),
        diff_cb_xgb=meta["p_cb"] - meta["p_xgb"],
        diff_cb_nn=meta["p_cb"] - meta["p_nn"],
        diff_xgb_nn=meta["p_xgb"] - meta["p_nn"],
        prod_cb_xgb=meta["p_cb"] * meta["p_xgb"],
        prod_cb_nn=meta["p_cb"] * meta["p_nn"],
        prod_xgb_nn=meta["p_xgb"] * meta["p_nn"],
    )[META_COLS]

    # 7. Meta-classifier, blended with the base average; the threshold is
    #    chosen on the training rows' blend, as in the original.
    meta_probs = meta.skb.apply(
        FitOn(
            LogisticRegression(
                C=1.0, penalty="l2", random_state=42, max_iter=1000
            )
        ),
        y=y,
    )
    blended = (
        BLEND_WEIGHT * meta_probs + (1.0 - BLEND_WEIGHT) * base_average
    ).to_frame("ensemble_prob")
    pred = blended.skb.apply(ThresholdSearch(), y=y)

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
