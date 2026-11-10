import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    KFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    """Learn grouped medians and the subsequent global fallback."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        grouped = pd.Series(self.medians_.reindex(keys).to_numpy(), index=X.index)
        self.fallback_ = X["Age"].fillna(grouped).median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        grouped = pd.Series(self.medians_.reindex(keys).to_numpy(), index=X.index)
        return X[["Age"]].assign(
            Age=X["Age"].fillna(grouped).fillna(self.fallback_)
        )


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
    """Internal validation, never the outer rows used to report accuracy."""

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
        # Preserve keys and feature schemas if fit-only branches are visited
        # during prediction. No validation labels are available in that mode.
        return {"X": X, "X_val": X, "y": None, "y_val": None}


class CatBoostStackingFeature(CatBoostClassifier):
    """Inner-validation probabilities on fit, holdout probabilities on score."""

    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))

    def fit_transform(self, X, y=None, *, eval_set, early_stopping_rounds):
        # GetXY already carved the rows. This only forwards them to CatBoost.
        self.fit(
            X,
            y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )
        return self.transform(eval_set[0])

    def transform(self, X):
        return pd.DataFrame({"p_cb": self.predict_proba(X)[:, 1]})


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


class NeuralStackingFeature(TransformerMixin, BaseEstimator):
    """Original torch training loop and validation-based scheduling.

    fit_transform emits probabilities on the inner validation rows. transform
    emits them for the rows it is given, which at score time is the holdout.
    """

    def __init__(self, random_state=42, batch_size=32, num_epochs=150):
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_epochs = num_epochs

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
            max(2, int(round(d ** 0.5 * 2))) for d in dimensions
        ]
        # Two output classes are explicit in the original architecture.
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

        # These accuracy calculations control internal training, not reporting.
        # Keep the final epoch, as the original does.
        for _ in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for start in range(0, train_cat.size(0), self.batch_size):
                batch = permutation[start:start + self.batch_size]
                optimizer.zero_grad()
                logits = self.model_(train_cat[batch], train_num[batch])
                loss = criterion(logits, train_y[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                predictions = self.model_(val_cat, val_num).argmax(dim=1)
                validation_accuracy = accuracy_score(
                    y_val, predictions.cpu().numpy()
                )
            scheduler.step(validation_accuracy)
        return self

    def fit_transform(self, X, y=None, *, X_val, y_val, cat_dims):
        self.fit(X, y, X_val=X_val, y_val=y_val, cat_dims=cat_dims)
        return self.transform(X_val)

    def transform(self, X):
        categorical, numeric = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            probabilities = torch.softmax(
                self.model_(categorical, numeric), dim=1
            )[:, 1].cpu().numpy()
        return pd.DataFrame({"p_nn": np.asarray(probabilities).ravel()})


def power_mean_ensemble(w1, w2, p, p1, p2):
    total_w = w1 + w2 + 1e-8
    w1_norm, w2_norm = w1 / total_w, w2 / total_w
    if abs(p) < 1e-3:
        return np.exp(w1_norm * np.log(p1) + w2_norm * np.log(p2))
    return np.power(
        w1_norm * np.power(p1, p) + w2_norm * np.power(p2, p),
        1.0 / p,
    )


class CalibratedPowerEnsemble(ClassifierMixin, BaseEstimator):
    """Per-fit OOF calibration and original joint numerical optimization."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        labels = np.asarray(y).ravel()
        cb_probs = np.asarray(X["p_cb"]).ravel()
        nn_probs = np.asarray(X["p_nn"]).ravel()
        cb_cal = np.zeros_like(cb_probs, dtype=float)
        nn_cal = np.zeros_like(nn_probs, dtype=float)

        # This is the original INTERNAL calibration algorithm, not the outer
        # scoring loop. All its rows come from the outer training partition.
        calibration_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in calibration_cv.split(cb_probs, labels):
            iso_cb = IsotonicRegression(out_of_bounds="clip")
            iso_nn = IsotonicRegression(out_of_bounds="clip")
            iso_cb.fit(cb_probs[train_idx], labels[train_idx])
            iso_nn.fit(nn_probs[train_idx], labels[train_idx])
            cb_cal[val_idx] = iso_cb.predict(cb_probs[val_idx])
            nn_cal[val_idx] = iso_nn.predict(nn_probs[val_idx])

        cb_cal = np.clip(cb_cal, 1e-6, 1.0 - 1e-6)
        nn_cal = np.clip(nn_cal, 1e-6, 1.0 - 1e-6)

        def objective(params):
            w1, p, threshold = params
            w1 = np.clip(w1, 0.0, 1.0)
            threshold = np.clip(threshold, 0.01, 0.99)
            probabilities = power_mean_ensemble(
                w1, 1.0 - w1, p, cb_cal, nn_cal
            )
            return -accuracy_score(
                labels, (probabilities >= threshold).astype(int)
            )

        result = minimize(
            objective,
            x0=[0.5, 1.0, 0.5],
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
        )
        w1, self.power_, threshold = result.x
        self.weight_ = np.clip(w1, 0.0, 1.0)
        self.threshold_ = np.clip(threshold, 0.01, 0.99)

        # The original optimized on its scored labels. For honest outer
        # prediction, learn deployment calibration maps on internal rows only.
        self.iso_cb_ = IsotonicRegression(out_of_bounds="clip").fit(
            cb_probs, labels
        )
        self.iso_nn_ = IsotonicRegression(out_of_bounds="clip").fit(
            nn_probs, labels
        )
        return self

    def predict(self, X):
        cb_probs = np.asarray(X["p_cb"]).ravel()
        nn_probs = np.asarray(X["p_nn"]).ravel()
        cb_cal = np.clip(self.iso_cb_.predict(cb_probs), 1e-6, 1.0 - 1e-6)
        nn_cal = np.clip(self.iso_nn_.predict(nn_probs), 1e-6, 1.0 - 1e-6)
        probabilities = power_mean_ensemble(
            self.weight_, 1.0 - self.weight_, self.power_, cb_cal, nn_cal
        )
        return (probabilities >= self.threshold_).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read without executing it on import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark raw target and features. Preserve the single stratified hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Learned imputations and frequencies use
    # outer-training rows only, fixing their original whole-table leakage.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = raw_title.map({
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }).fillna("Rare")
    title = title.where(raw_title.notna(), "Missing")

    age = X[["Pclass", "Sex", "Age"]].skb.apply(
        GroupAgeImputer(), no_wrap=True
    )
    ticket_frequency = X[["Ticket"]].skb.apply(
        TicketFrequency(), no_wrap=True
    )
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))

    features = X.assign(Title=title)
    features = features.assign(Age=age["Age"])
    features = features.assign(
        Sex_Pclass=features["Sex"].astype(str) + "_" + features["Pclass"].astype(str)
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    family_tier = features["FamilySize"].astype(str)
    family_tier = family_tier.where(
        features["FamilySize"] > 4, "SmallFamily"
    )
    family_tier = family_tier.where(
        features["FamilySize"] <= 4, "LargeFamily"
    )
    family_tier = family_tier.where(
        features["FamilySize"] != 1, "IsAlone"
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=features["Cabin"].notna().astype(int))
    features = features.assign(TicketFreq=ticket_frequency["TicketFreq"])
    features = features.assign(Embarked=features["Embarked"].fillna("Missing"))
    features = features.assign(Fare=fare["Fare"])

    features = features[CAT_COLS].astype(str).skb.concat(
        [features[NUM_COLS]], axis=1
    )[FEATURE_COLS]

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

    # The original used scored rows for stopping, scheduling, calibration, and
    # blend optimization. Retain these algorithms on an internal stratified 20%
    # split of the outer training rows. Both GetXY nodes share the fraction
    # and seed, so their rows stay aligned.
    pieces = features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_pieces = nn_matrix.skb.apply(GetXY(), y=y, no_wrap=True)
    fit_features = pieces["X"]
    val_features = pieces["X_val"]
    fit_y = pieces.get("y", y)
    val_y = pieces["y_val"]
    nn_fit = nn_pieces["X"]
    nn_val = nn_pieces["X_val"]

    # Each model node returns inner-validation probabilities from fit_transform
    # and holdout probabilities from transform.
    cb_probs = fit_features[FEATURE_COLS].skb.apply(
        CatBoostStackingFeature(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=CAT_COLS,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        no_wrap=True,
        y=fit_y,
        fit_transform_kwargs={
            "eval_set": (val_features[FEATURE_COLS], val_y),
            "early_stopping_rounds": 50,
        },
    )
    nn_probs = nn_fit.skb.apply(
        NeuralStackingFeature(random_state=42, batch_size=32, num_epochs=150),
        no_wrap=True,
        y=fit_y,
        fit_transform_kwargs={
            "X_val": nn_val,
            "y_val": val_y,
            "cat_dims": cat_dims,
        },
    )
    pred = cb_probs.skb.concat([nn_probs], axis=1).skb.apply(
        CalibratedPowerEnsemble(), y=val_y
    )

    # 4. Score the untouched outer hold-out, using the marked splitter.
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
