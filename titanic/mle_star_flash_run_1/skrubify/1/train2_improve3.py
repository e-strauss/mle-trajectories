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


class PeriodicNumericalEmbedding(nn.Module):
    """
    Learnable Periodic (Fourier) Numerical Embedding for continuous features.
    Transforms each scalar feature into a rich multi-scale periodic representation.
    """
    def __init__(self, num_features, token_dim, sigma=0.5):
        super().__init__()
        self.num_features = num_features
        self.token_dim = token_dim
        half_dim = token_dim // 2
        self.frequencies = nn.Parameter(torch.randn(num_features, half_dim) * sigma)
        self.linear = nn.Linear(half_dim * 2, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)

    def forward(self, x_num):
        # x_num: (batch_size, num_features)
        # x_proj: (batch_size, num_features, half_dim)
        x_proj = 2 * math.pi * x_num.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        fourier = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        out = self.linear(fourier)
        return self.layer_norm(out)


class FeatureSqueezeAndExcitation(nn.Module):
    """
    Feature-level Squeeze-and-Excitation module to dynamically recalibrate
    feature token importances per instance.
    """
    def __init__(self, num_tokens, reduction=4):
        super().__init__()
        reduced_dim = max(4, num_tokens // reduction)
        self.fc = nn.Sequential(
            nn.Linear(num_tokens, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, num_tokens),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, num_tokens, token_dim)
        # Squeeze along embedding dimension
        squeeze = x.mean(dim=-1)  # (batch_size, num_tokens)
        excitation = self.fc(squeeze).unsqueeze(-1)  # (batch_size, num_tokens, 1)
        return x * excitation


class FeatureGLU(nn.Module):
    """
    Feature-wise Gated Linear Unit for non-linear token transformation.
    """
    def __init__(self, token_dim):
        super().__init__()
        self.linear = nn.Linear(token_dim, token_dim * 2)

    def forward(self, x):
        h, gate = self.linear(x).chunk(2, dim=-1)
        return h * torch.sigmoid(gate)


class PairwiseInteractionPooling(nn.Module):
    """
    Computes pairwise dot-product interactions between all feature tokens.
    """
    def __init__(self, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        # Indices for the upper triangular pairwise interactions
        triu_indices = torch.triu_indices(num_tokens, num_tokens, offset=1)
        self.register_buffer("row_idx", triu_indices[0])
        self.register_buffer("col_idx", triu_indices[1])

    def forward(self, tokens):
        # tokens: (batch_size, num_tokens, token_dim)
        # Normalized dot products
        tokens_norm = F.normalize(tokens, p=2, dim=-1)
        interaction_matrix = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))
        interactions = interaction_matrix[:, self.row_idx, self.col_idx]
        return interactions


class TabularNeuralNet(nn.Module):
    """
    Tabular Gated Feature Network with periodic numerical embeddings,
    feature-wise gating (GLU + Squeeze-and-Excitation), and pairwise interaction pooling.
    """
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=[128, 64], token_dim=32, dropout=0.2):
        super().__init__()
        self.token_dim = token_dim
        self.num_cat = len(cat_dims)
        self.num_num = num_features
        self.total_tokens = self.num_cat + self.num_num

        # Categorical feature embeddings projected to token_dim
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(num_c, e_dim),
                    nn.Linear(e_dim, token_dim) if e_dim != token_dim else nn.Identity(),
                )
                for num_c, e_dim in zip(cat_dims, emb_dims)
            ]
        )

        # Numerical periodic Fourier embeddings
        if self.num_num > 0:
            self.num_embedding = PeriodicNumericalEmbedding(self.num_num, token_dim)
        else:
            self.num_embedding = None

        # Feature-wise Gated Linear Unit
        self.token_glu = FeatureGLU(token_dim)
        self.token_norm = nn.LayerNorm(token_dim)

        # Squeeze-and-Excitation across feature tokens
        self.se_module = FeatureSqueezeAndExcitation(self.total_tokens)

        # Pairwise interaction pooling
        if self.total_tokens > 1:
            self.interaction_pool = PairwiseInteractionPooling(self.total_tokens)
            num_interactions = self.total_tokens * (self.total_tokens - 1) // 2
        else:
            self.interaction_pool = None
            num_interactions = 0

        # Input dimension for final MLP: flattened tokens + pairwise interactions
        mlp_in_dim = (self.total_tokens * token_dim) + num_interactions

        # Classifier MLP head
        layers = []
        prev_dim = mlp_in_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        token_list = []

        # Process categorical features into tokens: (batch_size, 1, token_dim)
        if self.num_cat > 0:
            for i, emb_layer in enumerate(self.cat_embeddings):
                cat_tok = emb_layer(x_cat[:, i]).unsqueeze(1)
                token_list.append(cat_tok)

        # Process numerical features into tokens: (batch_size, num_num, token_dim)
        if self.num_num > 0 and self.num_embedding is not None:
            num_tok = self.num_embedding(x_num)
            token_list.append(num_tok)

        # Unified feature tokens: (batch_size, total_tokens, token_dim)
        tokens = torch.cat(token_list, dim=1) if len(token_list) > 1 else token_list[0]

        # Apply GLU gating with residual connection and LayerNorm
        tokens = self.token_norm(tokens + self.token_glu(tokens))

        # Dynamic feature importance recalibration
        tokens = self.se_module(tokens)

        # Extract pairwise interactions
        flat_tokens = tokens.flatten(start_dim=1)
        if self.interaction_pool is not None:
            interactions = self.interaction_pool(tokens)
            combined = torch.cat([flat_tokens, interactions], dim=1)
        else:
            combined = flat_tokens

        return self.network(combined)


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
