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


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
    """Learn grouped medians and the original's fallback median."""

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        grouped = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        self.fallback_ = X["Age"].fillna(grouped).median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        grouped = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X["Age"].fillna(grouped).fillna(self.fallback_).to_frame("Age")


class TicketFrequency(TransformerMixin, BaseEstimator):
    """Learn ticket frequencies independently of the imputers."""

    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).fillna(1).to_frame("TicketFreq")


class GetXY(TransformerMixin, BaseEstimator):
    """Reserve an early-stopping set from this fold's training rows only."""

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


class CrossLayer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim, bias=True)

    def forward(self, x0, xl):
        return x0 * self.linear(xl) + xl


class SwiGLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc_gate = nn.Linear(in_dim, out_dim)
        self.fc_val = nn.Linear(in_dim, out_dim)
        self.fc_out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_proj(x)
        normed = self.norm(x)
        gate = F.silu(self.fc_gate(normed))
        val = self.fc_val(normed)
        out = self.fc_out(gate * val)
        return residual + self.dropout(out)


class TabularNeuralNet(nn.Module):
    def __init__(
        self,
        cat_dims,
        emb_dims,
        num_features,
        hidden_dims=(128, 64),
        num_cross_layers=2,
        emb_dropout=0.1,
        deep_dropout=0.1,
        out_dim=2,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, d) for n, d in zip(cat_dims, emb_dims)]
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.in_dim = sum(emb_dims) + num_features
        self.input_norm = nn.LayerNorm(self.in_dim)
        self.cross_layers = nn.ModuleList(
            [CrossLayer(self.in_dim) for _ in range(num_cross_layers)]
        )

        deep_layers = []
        prev_dim = self.in_dim
        for h_dim in hidden_dims:
            deep_layers.append(SwiGLUBlock(prev_dim, h_dim, dropout=deep_dropout))
            prev_dim = h_dim
        self.deep_network = nn.Sequential(*deep_layers)
        self.skip_head = nn.Linear(self.in_dim, out_dim, bias=False)
        combined_dim = self.in_dim + prev_dim
        self.out_norm = nn.LayerNorm(combined_dim)
        self.head = nn.Linear(combined_dim, out_dim)

    def forward(self, x_cat, x_num):
        if len(self.embeddings) > 0 and x_cat is not None and x_cat.size(1) > 0:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat_emb = self.emb_dropout(torch.cat(embs, dim=1))
            if x_num is not None and x_num.size(1) > 0:
                x0 = torch.cat([x_cat_emb, x_num], dim=1)
            else:
                x0 = x_cat_emb
        else:
            x0 = x_num

        x0_norm = self.input_norm(x0)
        xl = x0_norm
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0_norm, xl)
        xd = self.deep_network(x0_norm)
        combined = torch.cat([xl, xd], dim=1)
        return self.head(self.out_norm(combined)) + self.skip_head(x0_norm)


class TorchProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """Only the neural model and training loop; encoding is in the plan."""

    def __init__(
        self,
        batch_size=32,
        num_epochs=150,
        learning_rate=1e-2,
        weight_decay=1e-4,
        scheduler_factor=0.5,
        scheduler_patience=10,
        random_state=42,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.random_state = random_state

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

    def fit(self, X, y, eval_set=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Vocabulary sizes are computed on the whole outer-training fold, which
        # includes the scheduler holdout. Recomputing them here would see only
        # the rows this fit call trains on.
        cat_dims = np.asarray(cat_dims, dtype=int).ravel().tolist()
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
            optimizer,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        train_cat, train_num = self._tensors(X)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        X_val, y_val = eval_set
        val_cat, val_num = self._tensors(X_val)

        # Keep all 150 epochs. The original scheduled on the scored hold-out and
        # did not restore a best checkpoint. The eval rows arrive from GetXY.
        for _ in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for i in range(0, train_cat.size(0), self.batch_size):
                batch_idx = permutation[i : i + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(train_cat[batch_idx], train_num[batch_idx])
                loss = criterion(outputs, train_y[batch_idx])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_preds = (
                    torch.argmax(self.model_(val_cat, val_num), dim=1)
                    .cpu()
                    .numpy()
                )
            scheduler.step(accuracy_score(np.asarray(y_val), val_preds))

        return self

    def predict_proba(self, X):
        x_cat, x_num = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(x_cat, x_num), dim=1).cpu().numpy()

    def predict(self, X):
        # This node feeds ensemble arithmetic, not a standalone scorer.
        return self.predict_proba(X)


def ensemble_predictions(cb_probs, nn_probs, mode):
    if mode == "fit":
        return None
    # Preserve column-1 extraction, equal weights, and the >= threshold.
    probs = 0.5 * cb_probs[:, 1] + 0.5 * nn_probs[:, 1]
    return (probs >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read; importing the file performs no I/O.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the RAW target and retain the original
    # stratified 80/20 hold-out with the same seed and held-out rows.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
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

    # The original learned these statistics on the full table. Independent
    # fitted operators now learn them on each fold's training rows only.
    age = X[["Pclass", "Sex", "Age"]].skb.apply(
        GroupedAgeImputer(), no_wrap=True
    )
    ticket_freq = X[["Ticket"]].skb.apply(TicketFrequency(), no_wrap=True)
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))

    features = X.assign(Title=title)
    features = features.assign(Age=age["Age"])
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
        (family_size <= 4)
        .map({True: "SmallFamily", False: "LargeFamily"})
        .where(family_size != 1, "IsAlone")
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=features["Cabin"].notna().astype(int))
    features = features.assign(TicketFreq=ticket_freq["TicketFreq"])
    features = features.assign(Embarked=features["Embarked"].fillna("Missing"))
    features = features.assign(Fare=fare["Fare"])

    categorical = features[CAT_COLS].astype(str)
    numeric = features[NUM_COLS]
    cb_features = categorical.skb.concat([numeric], axis=1)[FEATURE_COLS]

    # 3a. CatBoost -- preserve all settings and 50-round early stopping.
    # The original early-stopped on scored rows. Reserve 20% of the fold's
    # training rows instead; its original leaky score is not reproducible.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
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

    # 3b. Neural preprocessing -- sorted categories match LabelEncoder.
    # OrdinalEncoder forbids using a known category's code as unknown_value.
    # Encode unseen values as -1, then record their replacement by 0 to
    # reproduce the original mapping.get(value, 0) exactly.
    nn_cat = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    )
    nn_cat = nn_cat.replace(-1, 0)

    # Preserve float32 input to the separately fitted StandardScaler.
    nn_num = numeric.astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_cat.skb.concat([nn_num], axis=1)[FEATURE_COLS]
    
    cat_dims = nn_features[CAT_COLS].max(axis=0) + 1
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_pred = nn_parts["X"].skb.apply(
        TorchProbabilityClassifier(),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "eval_set": (nn_parts["X_val"], nn_parts["y_val"]),
            "cat_dims": cat_dims,
        },
    )

    # 3c. One ensemble, not separate scored variants. Gate arithmetic because
    # predictor nodes evaluate to fitted estimators in fit mode.
    pred = cb_pred.skb.apply_func(
        ensemble_predictions, nn_pred, skrub.eval_mode()
    )

    # 4. Score -- mark_as_X drives the single validation hold-out.
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
