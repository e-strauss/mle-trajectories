import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = [
    "Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"
]
META_COLS = [
    "p_cb", "p_nn", "rank_cb", "rank_nn", "diff_prob", "abs_diff_prob",
    "prod_prob", "power_mean_half", "power_mean_two", "entropy_cb",
    "entropy_nn", "entropy_mean",
]


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        medians = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        self.fallback_ = X["Age"].fillna(medians).median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        medians = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X.assign(Age=X["Age"].fillna(medians).fillna(self.fallback_))


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
    """Reserve part of this outer fold's training rows for stacking."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        parts = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        # Keep feature placeholders usable by downstream column selections.
        # The validation entries are consumed only during fitting.
        return {"X": X, "X_val": X, "y": None, "y_val": None}


class CatBoostStackingFeature(CatBoostClassifier):
    """Emit internal validation probabilities on fit, scored probabilities later."""

    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))

    def fit_transform(
        self, X, y=None, *, eval_set, early_stopping_rounds
    ):
        # The split itself is recorded by GetXY.
        self.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )
        return self.transform(eval_set[0])

    def transform(self, X):
        # Both probability blocks use positional indices. The meta-target is
        # likewise consumed positionally, preserving internal validation order.
        return pd.DataFrame({"p_cb": self.predict_proba(X)[:, 1]})


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


class NeuralStackingFeature(TransformerMixin, BaseEstimator):
    """The torch training loop, with explicitly supplied validation rows."""

    def __init__(
        self,
        batch_size=32,
        num_epochs=150,
        learning_rate=1e-2,
        weight_decay=1e-4,
        random_state=42,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _tensors(self, X):
        categorical = torch.tensor(
            X[CAT_COLS].to_numpy(copy=True),
            dtype=torch.long,
            device=self.device_,
        )
        numeric = torch.tensor(
            X[NUM_COLS].to_numpy(copy=True),
            dtype=torch.float32,
            device=self.device_,
        )
        return categorical, numeric

    def fit(self, X, y=None, *, X_val, y_val, cat_dims):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Vocabulary size comes from every outer training row, matching
        # len(LabelEncoder.classes_) on idx_train. The inner fit rows can miss
        # a category that still appears in the inner eval slice.
        cat_dims = [int(dim) for dim in np.asarray(cat_dims).ravel()]
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]
        self.model_ = TabularNeuralNet(
            cat_dims, emb_dims, num_features=len(NUM_COLS)
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
        x_cat, x_num = self._tensors(X)
        val_cat, val_num = self._tensors(X_val)
        target = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )

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
                validation_predictions = (
                    self.model_(val_cat, val_num).argmax(dim=1).cpu().numpy()
                )
            scheduler.step(accuracy_score(y_val, validation_predictions))
        return self

    def fit_transform(self, X, y=None, *, X_val, y_val, cat_dims):
        self.fit(X, y, X_val=X_val, y_val=y_val, cat_dims=cat_dims)
        return self.transform(X_val)

    def transform(self, X):
        self.model_.eval()
        x_cat, x_num = self._tensors(X)
        with torch.no_grad():
            probabilities = (
                torch.softmax(self.model_(x_cat, x_num), dim=1)[:, 1]
                .cpu()
                .numpy()
            )
        return pd.DataFrame({"p_nn": probabilities})


class CalibratedThresholdMetaModel(ClassifierMixin, BaseEstimator):
    def __init__(self, random_state=42):
        self.random_state = random_state

    def _make_model(self):
        base = LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 20),
            penalty="l2",
            solver="lbfgs",
            cv=3,
            scoring="accuracy",
            max_iter=1000,
            random_state=self.random_state,
        )
        return CalibratedClassifierCV(
            estimator=base, method="sigmoid", cv=3
        )

    def fit(self, X, y):
        target = np.asarray(y).ravel()
        self.classes_ = np.unique(target)
        inner_cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.random_state
        )
        oof_probabilities = np.zeros(len(target))

        # Intentional internal threshold-training CV, not an outer scoring loop.
        # Each call sees only the outer fold's training-derived rows.
        for train_indices, validation_indices in inner_cv.split(X, target):
            model = self._make_model()
            model.fit(X.iloc[train_indices], target[train_indices])
            oof_probabilities[validation_indices] = model.predict_proba(
                X.iloc[validation_indices]
            )[:, 1]

        fpr, tpr, thresholds = roc_curve(target, oof_probabilities)
        positives = np.sum(target == 1)
        negatives = np.sum(target == 0)
        accuracies = (
            tpr * positives + (1.0 - fpr) * negatives
        ) / len(target)
        self.threshold_ = thresholds[np.argmax(accuracies)]
        if np.isinf(self.threshold_) or self.threshold_ > 1.0:
            self.threshold_ = 0.5

        # The source evaluates OOF predictions after choosing a threshold on
        # their labels. For untouched outer rows, refit the same calibrated
        # learner on all internal meta-training rows.
        self.model_ = self._make_model().fit(X, target)
        return self

    def predict(self, X):
        probabilities = self.model_.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold_).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read without executing it on import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — raw target and the original stratified 80/20 hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = raw_title.map(
        {
            "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        }
    ).fillna("Rare").where(raw_title.notna(), "Missing")

    family_size = X["SibSp"] + X["Parch"] + 1
    family_tier = family_size.eq(1).map(
        {True: "IsAlone", False: "SmallFamily"}
    ).where(family_size.eq(1) | family_size.le(4), "LargeFamily")

    features = X.assign(
        Title=title,
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str),
        FamilySize=family_size,
        FamilyTier=family_tier,
        HasCabin=X["Cabin"].notna().astype(int),
        Embarked=X["Embarked"].fillna("Missing"),
    )

    # Independent learned statistics are separate nodes. Full-table imputation
    # and ticket-frequency leakage are fixed by fold-local fits.
    features = features.skb.apply(
        GroupedAgeImputer(), cols=["Pclass", "Sex", "Age"]
    )
    features = features.skb.apply(TicketFrequency(), cols=["Ticket"])
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Fare"]
    )
    categorical = features[CAT_COLS].astype(str)
    features = categorical.skb.concat([features[NUM_COLS]], axis=1)

    # LabelEncoder and StandardScaler were fit on every outer training row.
    # Learn them before the inner carve. Unknown categories map to 0, matching
    # the original mapping.get(category, 0).
    nn_categories = features[CAT_COLS].skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    ).clip(lower=0)
    nn_numbers = features[NUM_COLS].astype(np.float32).skb.apply(StandardScaler())
    nn_matrix = nn_categories.skb.concat([nn_numbers], axis=1)
    cat_dims = (nn_categories.max(axis=0) + 1).to_numpy()

    # 4. Recorded internal validation split. The original uses scored rows for
    # early stopping, NN scheduling, stacking, and threshold selection. Keep
    # these mechanisms on a 20% partition inside the outer training fold.
    # Both GetXY nodes share the fraction and seed, so their rows stay aligned.
    parts = features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_parts = nn_matrix.skb.apply(GetXY(), y=y, no_wrap=True)
    X_fit = parts["X"]
    y_fit = parts["y"]
    X_val = parts["X_val"]
    y_val = parts["y_val"]

    # Intermediate stacking transformers emit internal validation probabilities
    # on fit_transform and untouched outer probabilities on transform.
    # The CatBoost split remains visible in GetXY. Since this is a transformer,
    # fit_transform_kwargs forwards its eval_set and patience explicitly.
    cb_features = X_fit.skb.apply(
        CatBoostStackingFeature(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=CAT_COLS,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=y_fit,
        no_wrap=True,
        fit_transform_kwargs={
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 50,
        },
    )

    # 5. The network trains on the inner slice of a matrix encoded from every
    # outer training row. cat_dims is that full vocabulary, not the inner max.
    nn_features = nn_parts["X"].skb.apply(
        NeuralStackingFeature(),
        y=y_fit,
        no_wrap=True,
        fit_transform_kwargs={
            "X_val": nn_parts["X_val"],
            "y_val": y_val,
            "cat_dims": cat_dims,
        },
    )

    # 6. Recorded meta-features — preserve named columns and original order.
    p_cb = cb_features["p_cb"]
    p_nn = nn_features["p_nn"]
    eps = 1e-15
    clipped_cb = p_cb.clip(eps, 1.0 - eps)
    clipped_nn = p_nn.clip(eps, 1.0 - eps)
    difference = p_cb - p_nn
    mean_probability = 0.5 * (clipped_cb + clipped_nn)

    meta = cb_features.skb.concat([nn_features], axis=1)
    # These are the source's stateless ranks within the current prediction
    # batch (scipy.stats.rankdata / len), including average ranks for ties.
    # They neither learn a reference distribution nor inspect target labels.
    meta = meta.assign(rank_cb=p_cb.rank(method="average", pct=True))
    meta = meta.assign(rank_nn=p_nn.rank(method="average", pct=True))
    meta = meta.assign(diff_prob=difference)
    meta = meta.assign(abs_diff_prob=difference.abs())
    meta = meta.assign(prod_prob=p_cb * p_nn)
    meta = meta.assign(
        power_mean_half=((clipped_cb**0.5 + clipped_nn**0.5) / 2.0) ** 2.0
    )
    meta = meta.assign(
        power_mean_two=((clipped_cb**2.0 + clipped_nn**2.0) / 2.0) ** 0.5
    )
    meta = meta.assign(
        entropy_cb=-(
            clipped_cb * clipped_cb.skb.apply_func(np.log)
            + (1.0 - clipped_cb)
            * (1.0 - clipped_cb).skb.apply_func(np.log)
        )
    )
    meta = meta.assign(
        entropy_nn=-(
            clipped_nn * clipped_nn.skb.apply_func(np.log)
            + (1.0 - clipped_nn)
            * (1.0 - clipped_nn).skb.apply_func(np.log)
        )
    )
    meta = meta.assign(
        entropy_mean=-(
            mean_probability * mean_probability.skb.apply_func(np.log)
            + (1.0 - mean_probability)
            * (1.0 - mean_probability).skb.apply_func(np.log)
        )
    )
    meta = meta[META_COLS]

    # 7. Fit calibrated stacking and the original ROC threshold arithmetic
    # using only internal training-derived labels.
    pred = meta.skb.apply(CalibratedThresholdMetaModel(), y=y_val)

    # 8. Score — the splitter on mark_as_X drives outer validation.
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
