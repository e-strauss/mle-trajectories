import math

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS
TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}


def engineer(data):
    """Full-table features, including the rows the original later scores."""
    raw_title = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = (
        raw_title.map(TITLE_MAP).fillna("Rare").where(raw_title.notna(), "Missing")
    )
    features = data.assign(Title=title)
    grouped_age = features.groupby(["Pclass", "Sex"])["Age"].transform("median")
    features = features.assign(Age=features["Age"].fillna(grouped_age))
    features = features.assign(Age=features["Age"].fillna(features["Age"].median()))
    features = features.assign(
        Sex_Pclass=features["Sex"].astype(str) + "_" + features["Pclass"].astype(str)
    )
    features = features.assign(FamilySize=features["SibSp"] + features["Parch"] + 1)
    family_size = features["FamilySize"]
    family_tier = (
        family_size.le(4)
        .map({True: "SmallFamily", False: "LargeFamily"})
        .where(family_size.ne(1), "IsAlone")
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=features["Cabin"].notna().astype(int))
    ticket_counts = features["Ticket"].value_counts()
    features = features.assign(
        TicketFreq=features["Ticket"].map(ticket_counts).fillna(1)
    )
    features = features.assign(Embarked=features["Embarked"].fillna("Missing"))
    features = features.assign(
        Fare=features["Fare"].fillna(features["Fare"].median())
    )
    return features


def as_model_frame(frame):
    """Feature columns in the original order, with categoricals as strings."""
    selected = frame[FEATURE_COLS]
    return selected.assign(
        **{column: selected[column].astype(str) for column in CAT_COLS}
    )


def holdout_rows(df):
    indices = np.arange(len(df))
    _, idx_val, _, _ = train_test_split(
        indices,
        df["Survived"],
        test_size=0.2,
        random_state=42,
        stratify=df["Survived"],
    )
    return df.iloc[idx_val]


def fit_encoders(X):
    encoders = [
        LabelEncoder().fit(X[column].astype(str)) for column in CAT_COLS
    ]
    scaler = StandardScaler().fit(X[NUM_COLS].to_numpy().astype(np.float32))
    return encoders, scaler


def encode_matrices(X, encoders, scaler):
    columns = []
    for column, encoder in zip(CAT_COLS, encoders):
        mapping = {value: code for code, value in enumerate(encoder.classes_)}
        encoded = np.array(
            [mapping.get(value, 0) for value in X[column].astype(str).to_numpy()]
        )
        columns.append(encoded)
    categorical = np.stack(columns, axis=1).astype(np.int64)
    numeric = scaler.transform(X[NUM_COLS].to_numpy().astype(np.float32))
    return categorical, numeric


class NumericalTokenizer(nn.Module):
    def __init__(self, num_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_token) * (1.0 / math.sqrt(d_token)))
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))

    def forward(self, x_num):
        # x_num shape: (B, num_features)
        # Output shape: (B, num_features, d_token)
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TabularNeuralNet(nn.Module):
    def __init__(
        self,
        cat_dims,
        emb_dims,
        num_features,
        hidden_dims=[64, 32],
        d_token=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        out_dim=2,
    ):
        super().__init__()
        self.num_features = num_features
        self.cat_dims = cat_dims

        # Categorical Embeddings projected to unified token dimension
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(num_c, d_token) for num_c in cat_dims]
        )

        # Numerical Tokenizer
        if num_features > 0:
            self.num_tokenizer = NumericalTokenizer(num_features, d_token)
            self.num_skip = nn.Linear(num_features, out_dim)
        else:
            self.num_tokenizer = None
            self.num_skip = None

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        # Multi-Head Attention Transformer Backbone with Pre-Layer Normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Head from [CLS] representation
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, out_dim),
        )

    def forward(self, x_cat, x_num):
        batch_size = x_cat.size(0) if x_cat is not None and x_cat.nelement() > 0 else x_num.size(0)
        tokens = [self.cls_token.expand(batch_size, -1, -1)]

        if self.cat_embeddings and x_cat is not None and x_cat.size(1) > 0:
            cat_tokens = [
                emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)
            ]
            tokens.append(torch.cat(cat_tokens, dim=1))

        if self.num_tokenizer is not None and x_num is not None and x_num.size(1) > 0:
            num_tokens = self.num_tokenizer(x_num)
            tokens.append(num_tokens)

        # Contextual cross-feature representations
        x = torch.cat(tokens, dim=1)
        x = self.transformer(x)

        # Extract [CLS] token representation
        cls_rep = x[:, 0, :]
        logits = self.head(cls_rep)

        # Residual skip connection from raw numerical features
        if self.num_skip is not None and x_num is not None and x_num.size(1) > 0:
            logits = logits + self.num_skip(x_num)

        return logits


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Train the original network on the outer training rows.

    Categorical codes and the numeric scaler are fit on those rows only.
    Unseen categories map to code 0. The scored hold-out is the scheduler's
    validation slice, and the weights kept are the final epoch's.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y, *, X_val, y_val):
        y = np.asarray(y).ravel()
        y_val = np.asarray(y_val).ravel()
        self.classes_ = np.unique(y)
        self.encoders_, self.scaler_ = fit_encoders(X)
        train_cat, train_num = encode_matrices(X, self.encoders_, self.scaler_)
        val_cat, val_num = encode_matrices(X_val, self.encoders_, self.scaler_)

        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(self.random_state)
        dimensions = [len(encoder.classes_) for encoder in self.encoders_]
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
        train_cat_t = torch.tensor(
            train_cat, dtype=torch.long, device=self.device_
        )
        train_num_t = torch.tensor(
            train_num, dtype=torch.float32, device=self.device_
        )
        train_y = torch.tensor(y, dtype=torch.long, device=self.device_)
        val_cat_t = torch.tensor(val_cat, dtype=torch.long, device=self.device_)
        val_num_t = torch.tensor(
            val_num, dtype=torch.float32, device=self.device_
        )

        batch_size = 32
        for _epoch in range(150):
            self.model_.train()
            permutation = torch.randperm(train_cat_t.size(0))
            for start in range(0, train_cat_t.size(0), batch_size):
                batch = permutation[start : start + batch_size]
                optimizer.zero_grad()
                outputs = self.model_(train_cat_t[batch], train_num_t[batch])
                loss = criterion(outputs, train_y[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(val_cat_t, val_num_t)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)
            scheduler.step(val_acc)

        self.model_.eval()
        return self

    def predict(self, X):
        categorical, numeric = encode_matrices(X, self.encoders_, self.scaler_)
        x_cat = torch.tensor(
            categorical, dtype=torch.long, device=self.device_
        )
        x_num = torch.tensor(numeric, dtype=torch.float32, device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(x_cat, x_num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def _positive_class_probability(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        return array[:, 1]
    return array.reshape(-1)


class MeanProbabilityEnsemble(ClassifierMixin, BaseEstimator):
    """Threshold the equal-weight average of two positive-class probabilities."""

    def fit(self, X, y, nn_probability=None):
        cb_probability = _positive_class_probability(X)
        nn_probability = _positive_class_probability(nn_probability)
        if cb_probability.shape != nn_probability.shape:
            raise ValueError(
                "CatBoost and neural-net probabilities have different lengths: "
                f"{cb_probability.shape[0]} and {nn_probability.shape[0]}."
            )
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict(self, X, nn_probability=None):
        cb_probability = _positive_class_probability(X)
        nn_probability = _positive_class_probability(nn_probability)
        averaged = 0.5 * cb_probability + 0.5 * nn_probability
        return (averaged >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)
    features = engineer(data)

    holdout = features.skb.apply_func(holdout_rows)
    X_val = as_model_frame(holdout)
    y_val = holdout["Survived"]

    y = features["Survived"].skb.mark_as_y()
    X = features.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )
    X_model = as_model_frame(X)

    cb_probability = X_model.skb.apply(
        CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=y,
        fit_kwargs={
            "cat_features": CAT_COLS,
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 50,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    nn_probability = X_model.skb.apply(
        NeuralPositiveProbability(),
        y=y,
        fit_kwargs={"X_val": X_val, "y_val": y_val},
    )
    pred = cb_probability.skb.apply(
        MeanProbabilityEnsemble(),
        y=y,
        fit_kwargs={"nn_probability": nn_probability},
        predict_kwargs={"nn_probability": nn_probability},
    )

    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
