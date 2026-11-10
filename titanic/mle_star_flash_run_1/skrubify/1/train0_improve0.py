import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin"]


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    """Learn one grouped imputation and its subsequent global fallback."""

    def __init__(self, value_col, group_cols):
        self.value_col = value_col
        self.group_cols = group_cols

    def _group_fill(self, X):
        keys = X.set_index(list(self.group_cols)).index
        medians = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        valid_group = X[list(self.group_cols)].notna().all(axis=1)
        # pandas groupby.transform excludes rows with a missing grouping key,
        # even when their original value was present.
        return X[self.value_col].fillna(medians).where(valid_group)

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(list(self.group_cols))[
            self.value_col
        ].median()
        # The original takes the fallback median AFTER grouped imputation.
        self.fallback_ = self._group_fill(X).median()
        return self

    def transform(self, X):
        return X.assign(
            **{self.value_col: self._group_fill(X).fillna(self.fallback_)}
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Expose an inner validation split of this outer fold's training rows."""

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
            embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Keep the original torch loop; preprocessing and splitting remain visible."""

    def __init__(
        self,
        hidden_dims=(64, 32),
        batch_size=32,
        num_epochs=150,
        learning_rate=1e-2,
        weight_decay=1e-4,
        random_state=42,
    ):
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

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

    def fit(self, X, y, eval_set=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        dimensions = np.asarray(cat_dims, dtype=int).tolist()
        embedding_dimensions = [
            max(2, int(round(dimension**0.5 * 2))) for dimension in dimensions
        ]
        self.model_ = TabularNeuralNet(
            dimensions,
            embedding_dimensions,
            num_features=len(NUM_COLS),
            hidden_dims=self.hidden_dims,
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

        train_cat, train_num = self._tensors(X)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        X_val, y_val = eval_set
        val_cat, val_num = self._tensors(X_val)

        for epoch in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))

            for start in range(0, train_cat.size(0), self.batch_size):
                batch_indices = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(
                    train_cat[batch_indices], train_num[batch_indices]
                )
                loss = criterion(outputs, train_y[batch_indices])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(val_cat, val_num)
                val_predictions = torch.argmax(
                    val_outputs, dim=1
                ).cpu().numpy()
            scheduler.step(accuracy_score(y_val, val_predictions))

        # The original uses the final epoch, not a best-epoch checkpoint.
        return self

    def predict_proba(self, X):
        self.model_.eval()
        categorical, numeric = self._tensors(X)
        with torch.no_grad():
            outputs = self.model_(categorical, numeric)
            return torch.softmax(outputs, dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def ensemble_predictions(cb_probabilities, nn_probabilities, mode):
    # prediction_type="Probability" returns (n_samples, n_classes) in classes_
    # order. classes_ is [0, 1], so column 1 is the positive-class probability
    # the net's predict already returns.
    if mode == "fit":
        return None
    cb_probabilities = np.asarray(cb_probabilities, dtype=float)[:, 1]
    return (
        0.5 * cb_probabilities + 0.5 * nn_probabilities >= 0.5
    ).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing the file records, but does not execute, this read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — retain the raw target. This splitter exactly reproduces
    #    train_test_split(..., stratify=y, test_size=0.2, random_state=42).
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original feature order.
    title_raw = (
        X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Missing")
    )
    title = title_raw.replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )
    title = title.where(
        title.isin(["Mr", "Miss", "Mrs", "Master", "Missing"]), "Rare"
    )
    features = X.assign(Title=title)
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    features = features.assign(
        IsAlone=(features["FamilySize"] == 1).astype(int)
    )
    features = features.assign(HasCabin=features["Cabin"].notna().astype(int))
    features = features.assign(Embarked=features["Embarked"].fillna("Missing"))
    features = features[CAT_COLS + NUM_COLS]

    # Each independent grouped statistic is its own learned operator.
    # Unlike the original full-table imputation, these learn only from the
    # outer fold's training rows, removing preprocessing leakage.
    features = features.skb.apply(
        GroupMedianImputer("Age", ("Title", "Pclass")), no_wrap=True
    )
    features = features.skb.apply(
        GroupMedianImputer("Fare", ("Pclass",)), no_wrap=True
    )

    # 3a. CatBoost branch. The original early-stopped on the scored rows.
    # Move that eval set inside the training fold, retaining the 20% fraction,
    # 600 iterations and 50-round patience; the leaky score is not reproducible.
    cb_features = features.astype(
        {"Pclass": str, "Sex": str, "Embarked": str, "Title": str}
    )
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
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

    # 3b. Neural branch. OrdinalEncoder uses the same sorted string categories
    # as each LabelEncoder; unseen values are explicitly mapped to index zero.
    nn_categorical = features[CAT_COLS].astype(str).skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    )
    nn_categorical = nn_categorical.replace(-1, 0).astype(np.int64)
    nn_numeric = features[NUM_COLS].astype(np.float32).skb.apply(
        StandardScaler()
    )
    nn_features = nn_categorical.skb.concat([nn_numeric], axis=1)
    nn_features = nn_features[CAT_COLS + NUM_COLS]

    # Dimensions are fit-only statistics evaluated on the outer training rows,
    # including its inner eval subset, just as the encoders are fitted there.
    cat_dimensions = nn_categorical.max(axis=0) + 1

    # The original scheduler also observed the scored validation labels.
    # Expose an inner split instead; keep its patience, factor and all 150 epochs.
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_probability = nn_parts["X"].skb.apply(
        NeuralPositiveProbability(),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "eval_set": (nn_parts["X_val"], nn_parts["y_val"]),
            "cat_dims": cat_dimensions,
        },
    )

    # The two models produce ONE ensemble score, not separate scored variants.
    # Preserve positive-probability averaging and the inclusive 0.5 threshold.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 4. Score — the splitter on mark_as_X drives; no outer manual fold loop.
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
