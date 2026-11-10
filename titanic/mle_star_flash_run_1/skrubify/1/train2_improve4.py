import math

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
    """Learn the grouped medians and the fallback median independently per fold."""

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        grouped_fill = self._group_fill(X)
        # The original computes this median AFTER filling by group.
        self.fallback_ = grouped_fill.median()
        return self

    def _group_fill(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        medians = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X["Age"].fillna(medians)

    def transform(self, X):
        return self._group_fill(X).fillna(self.fallback_).to_frame("Age")


class TicketFrequency(TransformerMixin, BaseEstimator):
    """Learn ticket frequencies on the training rows only."""

    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).fillna(1).to_frame("TicketFreq")


class GetXY(TransformerMixin, BaseEstimator):
    """Expose a training-only validation split as recorded graph outputs."""

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


class NumericalTokenizer(nn.Module):
    def __init__(self, num_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(num_features, d_token) * (1.0 / math.sqrt(d_token))
        )
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))

    def forward(self, x_num):
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TabularNeuralNet(nn.Module):
    def __init__(
        self,
        cat_dims,
        emb_dims,
        num_features,
        hidden_dims=(64, 32),
        d_token=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        out_dim=2,
    ):
        super().__init__()
        self.num_features = num_features
        self.cat_dims = cat_dims
        # emb_dims and hidden_dims are unused in the original architecture too.
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(num_c, d_token) for num_c in cat_dims]
        )

        if num_features > 0:
            self.num_tokenizer = NumericalTokenizer(num_features, d_token)
            self.num_skip = nn.Linear(num_features, out_dim)
        else:
            self.num_tokenizer = None
            self.num_skip = None

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, out_dim),
        )

    def forward(self, x_cat, x_num):
        batch_size = (
            x_cat.size(0)
            if x_cat is not None and x_cat.nelement() > 0
            else x_num.size(0)
        )
        tokens = [self.cls_token.expand(batch_size, -1, -1)]

        if self.cat_embeddings and x_cat is not None and x_cat.size(1) > 0:
            cat_tokens = [
                emb(x_cat[:, i]).unsqueeze(1)
                for i, emb in enumerate(self.cat_embeddings)
            ]
            tokens.append(torch.cat(cat_tokens, dim=1))

        if self.num_tokenizer is not None and x_num is not None and x_num.size(1) > 0:
            tokens.append(self.num_tokenizer(x_num))

        x = self.transformer(torch.cat(tokens, dim=1))
        logits = self.head(x[:, 0, :])
        if self.num_skip is not None and x_num is not None and x_num.size(1) > 0:
            logits = logits + self.num_skip(x_num)
        return logits


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Only the torch model/training loop; preprocessing remains in the graph."""

    def __init__(
        self,
        random_state=42,
        batch_size=32,
        num_epochs=150,
        lr=1e-2,
        weight_decay=1e-4,
    ):
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay

    def _tensors(self, X):
        categorical = torch.tensor(
            X[CAT_COLS].to_numpy(),
            dtype=torch.long,
            device=self.device_,
        )
        numerical = torch.tensor(
            X[NUM_COLS].to_numpy(),
            dtype=torch.float32,
            device=self.device_,
        )
        return categorical, numerical

    def fit(self, X, y, X_val=None, y_val=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        cat_dims = list(cat_dims)
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]
        self.model_ = TabularNeuralNet(
            cat_dims, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
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

        # Keep all 150 epochs and the scheduler; the original does not restore
        # a best checkpoint or stop the neural-network training early.
        for _ in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for start in range(0, train_cat.size(0), self.batch_size):
                indices = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(train_cat[indices], train_num[indices])
                loss = criterion(outputs, train_y[indices])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(val_cat, val_num)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, predictions)
            scheduler.step(val_acc)

        self.model_.eval()
        return self

    def predict(self, X):
        cat, num = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(cat, num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def ensemble_predictions(cb_probability, nn_probability, mode):
    if mode == "fit":
        return None
    cb_probability = np.asarray(cb_probability, dtype=float)[:, 1]
    probabilities = 0.5 * cb_probability + 0.5 * nn_probability
    return (probabilities >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read, with no I/O or training during import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target. StratifiedShuffleSplit reproduces
    #    the original stratified train_test_split's exact hold-out rows.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    #    Replace the per-row title parser with recorded mapping/where operations.
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

    # The original learned age/fare medians and ticket counts on the whole
    # table. Separate fitted nodes now learn them from each fold's training rows.
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
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str)
    )
    features = features.assign(FamilySize=X["SibSp"] + X["Parch"] + 1)
    family_size = features["FamilySize"]
    family_tier = (
        pd.Series  # The actual values are constructed by recorded Series ops.
    )
    family_tier = family_size.astype(str).where(
        family_size == 1, "SmallFamily"
    )
    family_tier = family_tier.where(family_size != 1, "IsAlone")
    family_tier = family_tier.where(
        (family_size == 1) | (family_size <= 4), "LargeFamily"
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=X["Cabin"].notna().astype(int))
    features = features.assign(TicketFreq=ticket_frequency["TicketFreq"])
    features = features.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.assign(Fare=fare["Fare"])

    categorical = features[CAT_COLS].astype(str)
    numerical = features[NUM_COLS]
    cb_features = categorical.skb.concat([numerical], axis=1)

    # Sorted OrdinalEncoder categories reproduce the separate LabelEncoders.
    # Unknown categories receive 0, exactly as in the original mapping.get(v, 0).
    nn_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    )
    nn_categorical = nn_categorical.replace(-1, 0)
    nn_numerical = numerical.astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_categorical.skb.concat([nn_numerical], axis=1)
    nn_features = nn_features[CAT_COLS + NUM_COLS]

    # This statistic is consumed only during model initialization, so it is a
    # recorded fit argument.
    cat_dims = (nn_categorical.max().astype(int) + 1).tolist()

    # The original early-stopped CatBoost and scheduled the neural learning rate
    # using the scored hold-out itself. Instead, expose matching 20% validation
    # splits of the outer fold's OWN training rows. Neither model sees the
    # scored rows during training; the original leaky score cannot be preserved.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)

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
    nn_probability = nn_parts["X"].skb.apply(
        NeuralPositiveProbability(
            random_state=42,
            batch_size=32,
            num_epochs=150,
            lr=1e-2,
            weight_decay=1e-4,
        ),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "X_val": nn_parts["X_val"],
            "y_val": nn_parts["y_val"],
            "cat_dims": cat_dims,
        },
    )

    # Preserve positive-column probability averaging and the >= 0.5 threshold.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 4. Score -- the splitter on mark_as_X drives the single stratified hold-out.
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
