import lightgbm as lgb
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


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class TitleAgeImputer(TransformerMixin, BaseEstimator):
    """Learn only the title-specific age medians."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby("Title")["Age"].median()
        return self

    def transform(self, X):
        return X.assign(Age=X["Age"].fillna(X["Title"].map(self.medians_)))


class GetXY(TransformerMixin, BaseEstimator):
    """Expose an internal validation split of the current fold's training rows."""

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
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Keep the original torch training loop, with recorded validation inputs."""

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
        numerical = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        return categorical, numerical

    def fit(self, X, y, X_val=None, y_val=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = np.unique(y)

        dimensions = np.asarray(cat_dims, dtype=int).tolist()
        embedding_dimensions = [
            max(2, int(round(dimension**0.5 * 2))) for dimension in dimensions
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
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )

        for epoch in range(self.num_epochs):
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
            scheduler.step(accuracy_score(y_val, predictions))

        return self

    def predict_proba(self, X):
        self.model_.eval()
        categorical, numerical = self._tensors(X)
        with torch.no_grad():
            outputs = self.model_(categorical, numerical)
            return torch.softmax(outputs, dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def positive_probability(values):
    """Column 1 of a probability matrix, or a vector that is already that column."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        return values[:, 1]
    return values


def ensemble_predictions(cb_probability, lgb_probability, nn_probability, mode):
    # Predictor nodes contain estimators during fitting, not probability arrays.
    if mode == "fit":
        return None
    probability = (
        0.4 * positive_probability(cb_probability)
        + 0.4 * positive_probability(lgb_probability)
        + 0.2 * positive_probability(nn_probability)
    )
    return (probability >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read; importing this file does not read the CSV.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and raw design matrix. This reproduces the original
    #    single stratified 80/20 hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, with separate learned imputation nodes.
    features = X.assign(
        Title=X["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
        .fillna("Missing")
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    features = features.assign(
        IsAlone=(features["FamilySize"] == 1).astype(int)
    )
    features = features.assign(
        CabinDeck=features["Cabin"].str[0].fillna("Missing")
    )
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing")
    )
    features = features[FEATURE_COLS]

    # The original learned these medians before splitting. Learn them from each
    # fold's training rows instead, including the fallback median AFTER filling
    # ages by title.
    features = features.skb.apply(TitleAgeImputer(), no_wrap=True)
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Age", "Fare"]
    )

    # CatBoost: retain all hyperparameters and the 50-round stopping patience.
    cb_features = features.astype(
        {
            "Pclass": str,
            "Sex": str,
            "Embarked": str,
            "Title": str,
            "CabinDeck": str,
        }
    )

    # The original stopped on the scored hold-out. That leaks validation
    # information; use a stratified 20% portion of the outer training rows for
    # stopping instead. The original scored hold-out is never used in fitting.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    # cat_features is a fit argument so sklearn clone does not see it.
    # prediction_type="Probability" returns both classes; the blend takes column 1.
    cb_probability = cb_parts["X"].skb.apply(
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

    # LightGBM uses the original categorical representation and parameters.
    lgb_features = features.astype(
        {
            "Pclass": "category",
            "Sex": "category",
            "Embarked": "category",
            "Title": "category",
            "CabinDeck": "category",
        }
    )
    lgb_probability = lgb_features.skb.apply(
        lgb.LGBMModel(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=42,
            verbose=-1,
        ),
        y=y,
    )

    # Neural preprocessing remains separate from the torch model. OrdinalEncoder
    # reproduces per-column LabelEncoder ordering; unseen values still map to 0.
    nn_categorical = features[CAT_COLS].astype(str).skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    )
    nn_categorical = nn_categorical.replace(-1, 0).astype(np.int64)
    nn_numerical = features[NUM_COLS].astype(np.float32).skb.apply(
        StandardScaler()
    )
    nn_features = nn_categorical.skb.concat([nn_numerical], axis=1)
    nn_features = nn_features[FEATURE_COLS]

    # Vocabulary sizes are fit-only statistics, evaluated on outer training rows.
    cat_dims = nn_categorical.nunique()

    # The original scheduler also consulted the scored hold-out every epoch.
    # Preserve the scheduler and all 150 epochs, but expose an internal validation
    # split so no outer validation labels are consumed by training.
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_probability = nn_parts["X"].skb.apply(
        NeuralPositiveProbability(
            random_state=42, batch_size=32, num_epochs=150
        ),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "X_val": nn_parts["X_val"],
            "y_val": nn_parts["y_val"],
            "cat_dims": cat_dims,
        },
    )

    # Keep the original weighted positive probabilities and >= 0.5 decoding.
    # These three models form ONE ensemble, not three separately scored variants.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions,
        lgb_probability,
        nn_probability,
        skrub.eval_mode(),
    )

    # 4. Score. Only the outer splitter on mark_as_X drives validation.
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
