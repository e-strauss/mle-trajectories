import copy

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.optim.swa_utils import AveragedModel


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
    """Learn group medians and the subsequent global fallback per fit."""

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        group_values = self.group_medians_.reindex(
            pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        )
        filled = X["Age"].fillna(
            pd.Series(group_values.to_numpy(), index=X.index)
        )
        # The original computes this median AFTER group-based filling.
        self.fallback_ = filled.median()
        return self

    def transform(self, X):
        group_values = self.group_medians_.reindex(
            pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        )
        age = X["Age"].fillna(
            pd.Series(group_values.to_numpy(), index=X.index)
        ).fillna(self.fallback_)
        return age.to_frame(name="Age")


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).fillna(1).to_frame("TicketFreq")


class GetXY(TransformerMixin, BaseEstimator):
    """Split only this outer fold's training rows for internal validation."""

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
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class NeuralProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """Keep the torch training loop, SWA, and temperature calibration here."""

    def __init__(
        self,
        batch_size=32,
        num_epochs=120,
        swa_start_epoch=80,
        feature_dropout_rate=0.05,
        patience=25,
        random_state=42,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.swa_start_epoch = swa_start_epoch
        self.feature_dropout_rate = feature_dropout_rate
        self.patience = patience
        self.random_state = random_state

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

    def fit(self, X, y, *, X_val, y_val, cat_dims):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        cat_dims = [int(d) for d in cat_dims]
        emb_dims = [max(2, int(round(d**0.5 * 2))) for d in cat_dims]
        model = TabularNeuralNet(
            cat_dims, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)

        train_cat, train_num = self._tensors(X)
        val_cat, val_num = self._tensors(X_val)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        val_y = torch.tensor(
            np.asarray(y_val), dtype=torch.long, device=self.device_
        )
        criterion = nn.CrossEntropyLoss()

        steps_per_epoch = (
            train_cat.size(0) + self.batch_size - 1
        ) // self.batch_size
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=8e-3,
            steps_per_epoch=steps_per_epoch,
            epochs=self.num_epochs,
            pct_start=0.15,
            div_factor=10.0,
            final_div_factor=1e3,
        )

        swa_model = AveragedModel(model)
        best_val_loss = float("inf")
        best_model_weights = None
        patience_counter = 0

        for epoch in range(self.num_epochs):
            model.train()
            permutation = torch.randperm(
                train_cat.size(0), device=self.device_
            )

            for i in range(0, train_cat.size(0), self.batch_size):
                indices = permutation[i : i + self.batch_size]
                batch_cat = train_cat[indices]
                batch_num = train_num[indices].clone()
                batch_y = train_y[indices]

                if self.feature_dropout_rate > 0 and batch_num.size(1) > 0:
                    mask = (
                        torch.rand(batch_num.shape, device=self.device_)
                        >= self.feature_dropout_rate
                    ).float()
                    batch_num = batch_num * mask

                optimizer.zero_grad()
                outputs = model(batch_cat, batch_num)
                loss = criterion(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()

            if epoch >= self.swa_start_epoch:
                swa_model.update_parameters(model)

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(val_cat, val_num), val_y).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if (
                patience_counter >= self.patience
                and epoch >= self.swa_start_epoch + 15
            ):
                break

        # Preserve the original SWA selection, including its lack of update_bn().
        if epoch >= self.swa_start_epoch:
            self.model_ = swa_model
        else:
            self.model_ = model
            if best_model_weights is not None:
                self.model_.load_state_dict(best_model_weights)

        self.model_.eval()
        with torch.no_grad():
            val_logits = self.model_(val_cat, val_num)

        temperature = nn.Parameter(torch.ones(1, device=self.device_))
        temp_optimizer = torch.optim.LBFGS(
            [temperature], lr=0.01, max_iter=50
        )

        def temp_eval():
            temp_optimizer.zero_grad()
            scaled_logits = val_logits / torch.clamp(temperature, min=1e-3)
            loss = criterion(scaled_logits, val_y)
            loss.backward()
            return loss

        temp_optimizer.step(temp_eval)
        self.temperature_ = temperature.detach()
        return self

    def predict_proba(self, X):
        categorical, numerical = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(categorical, numerical)
            logits = logits / torch.clamp(self.temperature_, min=1e-3)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        # Thresholding belongs to the final recorded ensemble operation.
        return self.predict_proba(X)[:, 1]


def ensemble_predictions(cb_probability, nn_probability, mode):
    # prediction_type="Probability" is (n_samples, n_classes); column 1 is P(1).
    if mode == "fit":
        return None
    cb_probability = np.asarray(cb_probability, dtype=float)[:, 1]
    return (
        0.5 * cb_probability + 0.5 * nn_probability >= 0.5
    ).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and features. Preserve the original stratified
    #    80/20 outer hold-out and random seed.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, with separate estimators for learned
    #    statistics. Whole-table age/fare imputation and ticket frequencies
    #    leaked in the original; these now learn from training rows only.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = raw_title.map(
        {
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
        }
    ).fillna("Rare").where(raw_title.notna(), "Missing")

    age = X[["Pclass", "Sex", "Age"]].skb.apply(
        GroupedAgeImputer(), no_wrap=True
    )
    ticket_frequency = X[["Ticket"]].skb.apply(
        TicketFrequency(), no_wrap=True
    )
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))

    family_size = X["SibSp"] + X["Parch"] + 1
    # Use an aligned boolean Series, not the scalar False rejected by where().
    # Applying the size==1 branch last preserves the original if/elif order.
    family_tier = (
        (family_size <= 4)
        .replace({True: "SmallFamily", False: "LargeFamily"})
        .where(family_size != 1, "IsAlone")
    )

    features = X.assign(
        Title=title,
        Age=age["Age"],
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str),
        FamilySize=family_size,
        FamilyTier=family_tier,
        HasCabin=X["Cabin"].notna().astype(int),
        TicketFreq=ticket_frequency["TicketFreq"],
        Embarked=X["Embarked"].fillna("Missing"),
        Fare=fare["Fare"],
    )
    categorical = features[CAT_COLS].astype(str)
    numerical = features[NUM_COLS]
    cb_features = categorical.skb.concat([numerical], axis=1)

    # 4. CatBoost. The original early-stopped on the scored hold-out; that leak
    #    cannot be reproduced honestly. Keep all model settings and patience,
    #    using an explicit inner 20% stratified hold-out from training rows.
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

    # 5. Neural preprocessing is represented by separate estimator nodes.
    #    OrdinalEncoder sorts string vocabularies like LabelEncoder; unseen
    #    categories map to zero, matching the original validation mapping.
    nn_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    ).replace(-1, 0)
    nn_numerical = numerical.astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_categorical.skb.concat([nn_numerical], axis=1)

    # Fit-only vocabulary sizes preserve embedding sizes even when an inner
    # split omits a level. These statistics are evaluated per outer fold.
    cat_dims = (nn_categorical.max() + 1).tolist()

    # The original also used scored rows for NN checkpoint selection and
    # temperature calibration. Move both to explicit inner validation rows;
    # preserve its torch loop, SWA arithmetic, and calibration parameters.
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_probability = nn_parts["X"].skb.apply(
        NeuralProbabilityClassifier(
            batch_size=32,
            num_epochs=120,
            swa_start_epoch=80,
            feature_dropout_rate=0.05,
            patience=25,
            random_state=42,
        ),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "X_val": nn_parts["X_val"],
            "y_val": nn_parts["y_val"],
            "cat_dims": cat_dims,
        },
    )

    # 6. Preserve the probability average and >= 0.5 decoding. Skip prediction
    #    arithmetic in fit mode, when prediction nodes contain estimators.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 7. Score only the original ensemble; neither component was scored alone.
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
