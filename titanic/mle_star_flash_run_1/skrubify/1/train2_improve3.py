import math

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    """Learn the Pclass/Sex age medians; global fallback is a separate node."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        medians = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        # The original groupby.transform also leaves null grouping keys null.
        age = X["Age"].where(X[["Pclass", "Sex"]].notna().all(axis=1))
        return X.assign(Age=age.fillna(medians))


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X.assign(TicketFreq=X["Ticket"].map(self.counts_).fillna(1))


class GetXY(TransformerMixin, BaseEstimator):
    """Expose an inner validation split of the outer fold's training rows."""

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


class PeriodicNumericalEmbedding(nn.Module):
    def __init__(self, num_features, token_dim, sigma=0.5):
        super().__init__()
        self.num_features = num_features
        self.token_dim = token_dim
        half_dim = token_dim // 2
        self.frequencies = nn.Parameter(
            torch.randn(num_features, half_dim) * sigma
        )
        self.linear = nn.Linear(half_dim * 2, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)

    def forward(self, x_num):
        x_proj = (
            2 * math.pi * x_num.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        )
        fourier = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        return self.layer_norm(self.linear(fourier))


class FeatureSqueezeAndExcitation(nn.Module):
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
        excitation = self.fc(x.mean(dim=-1)).unsqueeze(-1)
        return x * excitation


class FeatureGLU(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.linear = nn.Linear(token_dim, token_dim * 2)

    def forward(self, x):
        h, gate = self.linear(x).chunk(2, dim=-1)
        return h * torch.sigmoid(gate)


class PairwiseInteractionPooling(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        indices = torch.triu_indices(num_tokens, num_tokens, offset=1)
        self.register_buffer("row_idx", indices[0])
        self.register_buffer("col_idx", indices[1])

    def forward(self, tokens):
        tokens_norm = F.normalize(tokens, p=2, dim=-1)
        interaction_matrix = torch.bmm(
            tokens_norm, tokens_norm.transpose(1, 2)
        )
        return interaction_matrix[:, self.row_idx, self.col_idx]


class TabularNeuralNet(nn.Module):
    def __init__(
        self,
        cat_dims,
        emb_dims,
        num_features,
        hidden_dims=(128, 64),
        token_dim=32,
        dropout=0.2,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_cat = len(cat_dims)
        self.num_num = num_features
        self.total_tokens = self.num_cat + self.num_num

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(num_c, e_dim),
                    nn.Linear(e_dim, token_dim)
                    if e_dim != token_dim
                    else nn.Identity(),
                )
                for num_c, e_dim in zip(cat_dims, emb_dims)
            ]
        )
        self.num_embedding = (
            PeriodicNumericalEmbedding(self.num_num, token_dim)
            if self.num_num > 0
            else None
        )
        self.token_glu = FeatureGLU(token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.se_module = FeatureSqueezeAndExcitation(self.total_tokens)

        if self.total_tokens > 1:
            self.interaction_pool = PairwiseInteractionPooling(self.total_tokens)
            num_interactions = self.total_tokens * (self.total_tokens - 1) // 2
        else:
            self.interaction_pool = None
            num_interactions = 0

        prev_dim = self.total_tokens * token_dim + num_interactions
        layers = []
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
        if self.num_cat > 0:
            for i, emb_layer in enumerate(self.cat_embeddings):
                token_list.append(emb_layer(x_cat[:, i]).unsqueeze(1))
        if self.num_num > 0 and self.num_embedding is not None:
            token_list.append(self.num_embedding(x_num))

        tokens = (
            torch.cat(token_list, dim=1)
            if len(token_list) > 1
            else token_list[0]
        )
        tokens = self.token_norm(tokens + self.token_glu(tokens))
        tokens = self.se_module(tokens)
        flat_tokens = tokens.flatten(start_dim=1)
        if self.interaction_pool is not None:
            combined = torch.cat(
                [flat_tokens, self.interaction_pool(tokens)], dim=1
            )
        else:
            combined = flat_tokens
        return self.network(combined)


class TorchPositiveProbability(ClassifierMixin, BaseEstimator):
    """Only the original torch model and training loop; preprocessing is external."""

    def __init__(
        self,
        random_state=42,
        batch_size=32,
        num_epochs=150,
        learning_rate=1e-2,
        weight_decay=1e-4,
    ):
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _tensors(self, X):
        return (
            torch.tensor(
                X[CAT_COLS].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=self.device_,
            ),
            torch.tensor(
                X[NUM_COLS].to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=self.device_,
            ),
        )

    def fit(self, X, y, X_val=None, y_val=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        cat_dims = np.asarray(cat_dims, dtype=int).tolist()
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

        train_cat, train_num = self._tensors(X)
        val_cat, val_num = self._tensors(X_val)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )

        # As in the original, train all 150 epochs without restoring a best epoch.
        for epoch in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for i in range(0, train_cat.size(0), self.batch_size):
                indices = permutation[i : i + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(train_cat[indices], train_num[indices])
                loss = criterion(outputs, train_y[indices])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(val_cat, val_num)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_val, val_preds))

        self.model_.eval()
        return self

    def predict(self, X):
        cat, num = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(cat, num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def ensemble_predictions(cb_probs, nn_probs, mode):
    if mode == "fit":
        return None
    cb_probs = np.asarray(cb_probs, dtype=float)[:, 1]
    return (0.5 * cb_probs + 0.5 * nn_probs >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records the read without executing it.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — retain the raw target and the exact stratified 80/20
    #    outer hold-out used by train_test_split in the original.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and independent learned preprocessing.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = (
        raw_title.map(
            {
                "Mr": "Mr",
                "Miss": "Miss",
                "Mrs": "Mrs",
                "Master": "Master",
                "Mlle": "Miss",
                "Ms": "Miss",
                "Mme": "Mrs",
            }
        )
        .fillna("Rare")
        .where(raw_title.notna(), "Missing")
    )
    features = X.assign(Title=title)

    # These statistics originally used the complete table. Learn them only
    # from the outer fold's training rows, fixing preprocessing leakage.
    features = features.skb.apply(
        GroupAgeImputer(), cols=["Pclass", "Sex", "Age"]
    )
    features = features.skb.apply(
        SimpleImputer(strategy="median", keep_empty_features=True), cols=["Age"]
    )
    features = features.assign(
        Sex_Pclass=features["Sex"].astype(str)
        + "_"
        + features["Pclass"].astype(str)
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    family_tier = (
        features["FamilySize"]
        .map({1: "IsAlone"})
        .fillna("SmallFamily")
        .where(features["FamilySize"] <= 4, "LargeFamily")
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=features["Cabin"].notna().astype(int))
    features = features.skb.apply(TicketFrequency())
    features = features.assign(Embarked=features["Embarked"].fillna("Missing"))
    features = features.skb.apply(
        SimpleImputer(strategy="median", keep_empty_features=True), cols=["Fare"]
    )

    categorical = features[CAT_COLS].astype(str)
    features = categorical.skb.concat([features[NUM_COLS]], axis=1)
    features = features[FEATURE_COLS]

    # The original used the scored hold-out for CatBoost early stopping and
    # the neural learning-rate scheduler. Instead expose a stratified 20%
    # inner eval split of the outer TRAINING rows, preserving all training limits.
    cb_parts = features.skb.apply(GetXY(), y=y, no_wrap=True)
    cb_pred = cb_parts["X"].skb.apply(
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

    # OrdinalEncoder has LabelEncoder's sorted ordering for these string inputs.
    # Its unknown sentinel must NOT collide with a known-category code.
    # Encode unseen values as -1, then explicitly map them to zero to preserve
    # the original mapping.get(value, 0) behavior.
    nn_encoded = features.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        ),
        cols=CAT_COLS,
    )
    nn_categorical = nn_encoded[CAT_COLS].replace(-1, 0)
    nn_numeric = nn_encoded[NUM_COLS].astype(np.float32)
    nn_numeric = nn_numeric.skb.apply(StandardScaler())
    nn_features = nn_categorical.skb.concat([nn_numeric], axis=1)
    nn_features = nn_features[FEATURE_COLS]

    # Vocabulary sizes are fit-only statistics, evaluated on outer-training
    # rows and passed directly to the neural estimator.
    cat_dims = nn_features[CAT_COLS].max(axis=0) + 1
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_pred = nn_parts["X"].skb.apply(
        TorchPositiveProbability(
            random_state=42,
            batch_size=32,
            num_epochs=150,
            learning_rate=1e-2,
            weight_decay=1e-4,
        ),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "X_val": nn_parts["X_val"],
            "y_val": nn_parts["y_val"],
            "cat_dims": cat_dims,
        },
    )

    # Preserve positive-probability averaging and the original >= 0.5 tie rule.
    # Prediction arithmetic must be gated because fit-mode nodes are estimators.
    pred = cb_pred.skb.apply_func(
        ensemble_predictions, nn_pred, skrub.eval_mode()
    )

    # 4. Score the single original ensemble, not its individual components.
    #    The outer splitter is declared only on mark_as_X.
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
