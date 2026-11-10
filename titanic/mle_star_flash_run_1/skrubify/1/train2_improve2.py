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


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    """Learn grouped Age medians and the subsequent fallback median."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        filled = X["Age"].fillna(self._group_values(X))
        self.fallback_ = filled.median()
        return self

    def _group_values(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        return pd.Series(self.medians_.reindex(keys).to_numpy(), index=X.index)

    def transform(self, X):
        age = X["Age"].fillna(self._group_values(X)).fillna(self.fallback_)
        return X.assign(Age=age)


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X.assign(TicketFreq=X["Ticket"].map(self.counts_).fillna(1))


class GetXY(TransformerMixin, BaseEstimator):
    """An inner validation split of the outer fold's training rows."""

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


class TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, ffn_factor=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * ffn_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * ffn_factor, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.dropout1(attn_out)
        return x + self.ffn(self.norm2(x))


class TabularNeuralNet(nn.Module):
    def __init__(
        self,
        cat_dims=None,
        num_features=0,
        d_token=32,
        n_blocks=3,
        n_heads=4,
        ffn_factor=2,
        dropout=0.1,
        out_dim=2,
    ):
        super().__init__()
        cat_dims = cat_dims or []
        if d_token % n_heads != 0:
            d_token = ((d_token // n_heads) + 1) * n_heads

        self.d_token = d_token
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(num_c, d_token) for num_c in cat_dims]
        )
        self.num_embeddings = nn.ModuleList(
            [nn.Linear(1, d_token) for _ in range(num_features)]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.01)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    ffn_factor=ffn_factor,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, out_dim),
        )

    def forward(self, x_cat=None, x_num=None):
        batch_size = x_cat.size(0) if x_cat is not None else x_num.size(0)
        tokens = [self.cls_token.expand(batch_size, -1, -1)]

        if x_cat is not None and len(self.cat_embeddings) > 0:
            tokens.append(
                torch.stack(
                    [
                        embedding(x_cat[:, i])
                        for i, embedding in enumerate(self.cat_embeddings)
                    ],
                    dim=1,
                )
            )
        if x_num is not None and len(self.num_embeddings) > 0:
            tokens.append(
                torch.stack(
                    [
                        embedding(x_num[:, i : i + 1])
                        for i, embedding in enumerate(self.num_embeddings)
                    ],
                    dim=1,
                )
            )

        x = torch.cat(tokens, dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])


class TorchPositiveProbability(ClassifierMixin, BaseEstimator):
    """The original torch model, training loop, and probability output."""

    def fit(self, X, y, X_val=None, y_val=None, cat_dims=None):
        np.random.seed(42)
        torch.manual_seed(42)

        self.classes_ = np.unique(y)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = TabularNeuralNet(
            cat_dims=[int(n) for n in cat_dims],
            num_features=len(NUM_COLS),
            d_token=32,
            n_heads=4,
            n_blocks=2,
            dropout=0.1,
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-3, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        train_cat, train_num = self._tensors(X)
        val_cat, val_num = self._tensors(X_val)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )

        for epoch in range(100):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))

            for i in range(0, train_cat.size(0), 32):
                batch_indices = permutation[i : i + 32]
                optimizer.zero_grad()
                outputs = self.model_(
                    train_cat[batch_indices], train_num[batch_indices]
                )
                loss = criterion(outputs, train_y[batch_indices])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(val_cat, val_num)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                validation_accuracy = accuracy_score(y_val, predictions)
            scheduler.step(validation_accuracy)

        self.model_.eval()
        return self

    def _tensors(self, X):
        return (
            torch.tensor(
                X[CAT_COLS].to_numpy(),
                dtype=torch.long,
                device=self.device_,
            ),
            torch.tensor(
                X[NUM_COLS].to_numpy(),
                dtype=torch.float32,
                device=self.device_,
            ),
        )

    def predict(self, X):
        cat, num = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(cat, num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def ensemble_predictions(cb_probabilities, nn_probabilities, mode):
    if mode == "fit":
        return None
    cb_probabilities = np.asarray(cb_probabilities, dtype=float)[:, 1]
    probabilities = 0.5 * cb_probabilities + 0.5 * nn_probabilities
    return (probabilities >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the read; importing this module does not read data.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target. StratifiedShuffleSplit reproduces
    #    the original stratified train_test_split and its validation rows.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    #    Age medians, Fare medians, and ticket counts originally used the full
    #    table. They now learn only from the outer training fold, fixing leakage.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = raw_title.replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )
    title = title.where(
        title.isin(["Mr", "Miss", "Mrs", "Master"]), "Rare"
    )
    title = title.where(raw_title.notna(), "Missing")
    features = X.assign(Title=title)

    features = features.skb.apply(GroupAgeImputer())
    features = features.assign(
        Sex_Pclass=features["Sex"].astype(str)
        + "_"
        + features["Pclass"].astype(str)
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    family_size = features["FamilySize"]
    family_tier = (
        family_size.astype(str)
        .where(family_size == 1, "SmallFamily")
        .where(family_size <= 4, "LargeFamily")
        .where(family_size != 1, "IsAlone")
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(
        HasCabin=features["Cabin"].notna().astype(int)
    )
    features = features.skb.apply(TicketFrequency())
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing")
    )
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Fare"]
    )

    # Preserve the original categorical and numeric feature order.
    features = features[CAT_COLS + NUM_COLS]
    categorical = features[CAT_COLS].astype(str)
    cb_features = categorical.skb.concat([features[NUM_COLS]], axis=1)

    # CatBoost originally early-stopped on the scored hold-out. Retain its
    # patience and iterations, but carve an eval set from training rows instead.
    # Use the original split's fraction, seed, and stratification for this split.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    cb_probabilities = cb_parts["X"].skb.apply(
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

    # OrdinalEncoder reproduces the sorted per-column LabelEncoder codes.
    # Its unknown sentinel must NOT overlap a learned code: encode unknowns as
    # -1, then replace that sentinel with the original fallback code 0.
    nn_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    )
    nn_categorical = nn_categorical.replace(-1, 0)

    nn_numeric = (
        features[NUM_COLS]
        .astype(np.float32)
        .skb.apply(StandardScaler())
    )
    nn_features = nn_categorical.skb.concat([nn_numeric], axis=1)
    nn_features = nn_features[CAT_COLS + NUM_COLS]

    # Embedding sizes are fit-only statistics recorded downstream of the
    # training-fold encoder.
    cat_dims = nn_categorical.max().add(1).astype(int).tolist()

    # The neural scheduler originally consulted the scored hold-out as well.
    # Retain all 100 epochs and scheduler settings, but use an inner eval set.
    # Encoding and scaling remain fitted on the outer training rows.
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_probabilities = nn_parts["X"].skb.apply(
        TorchPositiveProbability(),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "X_val": nn_parts["X_val"],
            "y_val": nn_parts["y_val"],
            "cat_dims": cat_dims,
        },
    )

    # Preserve positive-class probability averaging and the >= 0.5 threshold.
    pred = cb_probabilities.skb.apply_func(
        ensemble_predictions, nn_probabilities, skrub.eval_mode()
    )

    # 4. Score the original ensemble, not its components independently.
    #    No cv= here: the splitter on mark_as_X drives validation.
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
