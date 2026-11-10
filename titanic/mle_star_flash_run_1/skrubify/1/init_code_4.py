import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

CAT_COLS = ["Pclass", "Sex", "Embarked", "Title"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GetXY(TransformerMixin, BaseEstimator):
    """Carve the scheduler's eval set from this fold's training rows."""

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
    def __init__(
        self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_c, e_dim)
                for num_c, e_dim in zip(cat_dims, emb_dims)
            ]
        )
        in_dim = sum(emb_dims) + num_features

        layers = []
        prev_dim = in_dim
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
            emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embs + [x_num], dim=1))


class TorchTabularClassifier(ClassifierMixin, BaseEstimator):
    """The original torch training loop, rerun on each fold's training rows."""

    def __init__(
        self,
        num_epochs=150,
        batch_size=32,
        learning_rate=1e-2,
        weight_decay=1e-4,
        random_state=42,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y, eval_set=None, cat_dims=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Cardinalities come from the outer training fold, passed in so the
        # scheduler split below does not shrink the embedding tables.
        cat_dims = [int(dim) for dim in np.asarray(cat_dims)]
        emb_dims = [
            max(2, int(round(dim**0.5 * 2))) for dim in cat_dims
        ]
        self.model_ = TabularNeuralNet(
            cat_dims, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)

        # X is already the inner training slice. The scheduler watches eval_set,
        # which GetXY carved from the outer training rows.
        X_val, y_val = eval_set
        train_cat = torch.tensor(
            X[CAT_COLS].to_numpy(),
            dtype=torch.long,
            device=self.device_,
        )
        train_num = torch.tensor(
            X[NUM_COLS].to_numpy(),
            dtype=torch.float32,
            device=self.device_,
        )
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        val_cat = torch.tensor(
            X_val[CAT_COLS].to_numpy(),
            dtype=torch.long,
            device=self.device_,
        )
        val_num = torch.tensor(
            X_val[NUM_COLS].to_numpy(),
            dtype=torch.float32,
            device=self.device_,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        for _ in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))

            for i in range(0, train_cat.size(0), self.batch_size):
                batch = permutation[i : i + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(train_cat[batch], train_num[batch])
                loss = criterion(outputs, train_y[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_preds = (
                    self.model_(val_cat, val_num)
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )
            scheduler.step(accuracy_score(y_val, val_preds))

        # As in the original, retain the final epoch, not the best epoch.
        return self

    def predict(self, X):
        x_cat = torch.tensor(
            X[CAT_COLS].to_numpy(),
            dtype=torch.long,
            device=self.device_,
        )
        x_num = torch.tensor(
            X[NUM_COLS].to_numpy(),
            dtype=torch.float32,
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            # Preserve the original's positional argmax prediction.
            return self.model_(x_cat, x_num).argmax(dim=1).cpu().numpy()


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this file records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target. StratifiedShuffleSplit reproduces
    #    the original stratified train_test_split's exact outer hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and fold-fitted preprocessing.
    features = X.assign(
        Title=X["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
        .fillna("Unknown"),
        FamilySize=X["SibSp"] + X["Parch"] + 1,
        Embarked=X["Embarked"].fillna("Missing"),
    )

    # Median imputation, categorical vocabularies, and scaling were originally
    # fitted before splitting. Fit each on training rows only to remove leakage.
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Age", "Fare"]
    )

    # OrdinalEncoder on strings matches LabelEncoder's sorted per-column codes.
    # Fold-only fitting introduces unseen validation categories; map those to
    # code 0 so they use an existing embedding rather than an invalid index.
    categorical = (
        features[CAT_COLS]
        .astype(str)
        .skb.apply(
            OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
        )
        .replace(-1, 0)
        .astype(np.int64)
    )
    numerical = (
        features[NUM_COLS]
        .astype(np.float32)
        .skb.apply(StandardScaler())
    )
    design = categorical.skb.concat([numerical], axis=1)[FEATURE_COLS]

    # Embedding widths follow the outer training vocabulary. Measured here, on
    # the whole outer training fold, so the scheduler split does not drop a
    # category that only appears in that inner eval slice.
    cat_dims = (design[CAT_COLS].max(axis=0) + 1).to_numpy()

    # The original stepped ReduceLROnPlateau on the scored hold-out. Keep the
    # scheduler, the patience, and all 150 epochs, but carve a stratified 20%
    # eval set from the outer training rows. The scored rows are never used.
    parts = design.skb.apply(
        GetXY(test_size=0.2, random_state=42), y=y, no_wrap=True
    )
    X_fit = parts["X"]
    y_fit = parts.get("y", y)
    X_val = parts["X_val"]
    y_val = parts["y_val"]
    pred = X_fit.skb.apply(
        TorchTabularClassifier(
            num_epochs=150,
            batch_size=32,
            learning_rate=1e-2,
            weight_decay=1e-4,
            random_state=42,
        ),
        y=y_fit,
        fit_kwargs={
            "eval_set": (X_val, y_val),
            "cat_dims": cat_dims,
        },
    )

    # 4. Score — the splitter on mark_as_X drives the single outer hold-out.
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
