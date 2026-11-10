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
from sklearn.preprocessing import LabelEncoder, StandardScaler


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
    """Learn the grouped medians and the subsequent global fallback separately."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        grouped = X.set_index(["Pclass", "Sex"]).index.map(self.medians_)
        filled = X["Age"].fillna(pd.Series(grouped, index=X.index))
        self.fallback_ = filled.median()
        return self

    def transform(self, X):
        grouped = X.set_index(["Pclass", "Sex"]).index.map(self.medians_)
        age = X["Age"].fillna(pd.Series(grouped, index=X.index))
        return age.fillna(self.fallback_).to_frame("Age")


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).fillna(1).to_frame("TicketFreq")


class CategoryEncoder(TransformerMixin, BaseEstimator):
    """The original LabelEncoder convention, including unknown -> 0."""

    def fit(self, X, y=None):
        self.encoders_ = {
            col: LabelEncoder().fit(X[col].astype(str)) for col in X.columns
        }
        return self

    def transform(self, X):
        return pd.DataFrame(
            {
                col: X[col].astype(str).map(
                    {value: i for i, value in enumerate(encoder.classes_)}
                ).fillna(0).astype(np.int64)
                for col, encoder in self.encoders_.items()
            },
            index=X.index,
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Reserve validation rows from the current outer fold's training rows."""

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


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Only the torch model and training loop; preprocessing stays in the DAG."""

    def __init__(
        self,
        cat_cols,
        num_cols,
        batch_size=32,
        num_epochs=150,
        random_state=42,
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_state = random_state

    def fit(self, X, y, eval_X=None, eval_y=None):
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # The original helper resets torch to 42 for EVERY ablation fit.
        torch.manual_seed(self.random_state)

        # Both tables have already passed through the same fitted encoder.
        # Taking the maximum across them recovers its complete vocabulary size.
        dimensions = (
            pd.concat([X[self.cat_cols], eval_X[self.cat_cols]])
            .max()
            .astype(int)
            .to_numpy()
            + 1
        ).tolist()
        embedding_dimensions = [
            max(2, int(round(dimension**0.5 * 2)))
            for dimension in dimensions
        ]
        self.model_ = TabularNeuralNet(
            dimensions, embedding_dimensions, len(self.num_cols)
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )
        cat_train = torch.tensor(
            X[self.cat_cols].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        num_train = torch.tensor(
            X[self.num_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        target_train = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        cat_val = torch.tensor(
            eval_X[self.cat_cols].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        num_val = torch.tensor(
            eval_X[self.num_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )

        for epoch in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(cat_train.size(0))
            for start in range(0, cat_train.size(0), self.batch_size):
                indices = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(cat_train[indices], num_train[indices])
                loss = criterion(outputs, target_train[indices])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                predictions = self.model_(cat_val, num_val).argmax(dim=1)
                validation_accuracy = accuracy_score(
                    eval_y, predictions.cpu().numpy()
                )
            scheduler.step(validation_accuracy)

        self.model_.eval()
        return self

    def predict(self, X):
        cat = torch.tensor(
            X[self.cat_cols].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        num = torch.tensor(
            X[self.num_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(cat, num), dim=1)[:, 1].cpu().numpy()


def blend_predictions(cb_probability, nn_probability, weight, mode):
    # prediction_type="Probability" is (n_samples, n_classes); column 1 is P(1).
    if mode == "fit":
        return None
    cb_probability = np.asarray(cb_probability, dtype=float)[:, 1]
    return (
        weight * cb_probability + (1.0 - weight) * nn_probability >= 0.5
    ).astype(int)


def add_interactions(frame):
    family_size = frame["FamilySize"]
    family_tier = (
        family_size.astype(str)
        .where(family_size != 1, "IsAlone")
        .where((family_size == 1) | (family_size > 4), "SmallFamily")
        .where((family_size == 1) | (family_size <= 4), "LargeFamily")
    )
    # NaN follows the original final "else", rather than becoming a string NaN.
    family_tier = family_tier.where(family_size.notna(), "LargeFamily")
    frequency = frame[["Ticket"]].skb.apply(TicketFrequency())
    return frame.assign(
        Sex_Pclass=frame["Sex"].astype(str) + "_" + frame["Pclass"].astype(str),
        FamilyTier=family_tier,
        TicketFreq=frequency["TicketFreq"],
    )


def make_probabilities(features, cat_cols, num_cols, y):
    ordered = features[cat_cols + num_cols]
    categorical_strings = ordered[cat_cols].astype(str)
    cb_features = categorical_strings.skb.concat(
        [ordered[num_cols]], axis=1
    )[cat_cols + num_cols]

    # The original used the scored hold-out for both CatBoost early stopping
    # and the NN learning-rate scheduler. That leaks validation information.
    # Preserve patience/epochs, but reserve a stratified 20% of outer TRAIN
    # for both mechanisms. The outer scored rows are never used for training.
    cb_split = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    cb_probability = cb_split["X"].skb.apply(
        CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_split.get("y", y),
        fit_kwargs={
            "cat_features": cat_cols,
            "eval_set": (cb_split["X_val"], cb_split["y_val"]),
            "early_stopping_rounds": 50,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )

    # Keep LabelEncoder's sorted classes and unknown-category fallback, and
    # cast numeric FEATURES to float32 before fitting StandardScaler.
    encoded_categories = ordered[cat_cols].skb.apply(CategoryEncoder())
    scaled_numbers = (
        ordered[num_cols]
        .astype(np.float32)
        .skb.apply(StandardScaler())
    )
    nn_features = encoded_categories.skb.concat(
        [scaled_numbers], axis=1
    )[cat_cols + num_cols]
    nn_split = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_probability = nn_split["X"].skb.apply(
        NeuralPositiveProbability(cat_cols=cat_cols, num_cols=num_cols),
        y=nn_split.get("y", y),
        fit_kwargs={
            "eval_X": nn_split["X_val"],
            "eval_y": nn_split["y_val"],
        },
    )
    return cb_probability, nn_probability


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this file records, but does not read, the CSV.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the RAW target. This splitter reproduces the
    # original stratified train_test_split's exact outer hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded features. Statistics originally learned before the split now
    # have individual fitted nodes, preventing full-table preprocessing leakage.
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

    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))
    common = X.assign(
        Title=title,
        FamilySize=X["SibSp"] + X["Parch"] + 1,
        HasCabin=X["Cabin"].notna().astype(int),
        Embarked=X["Embarked"].fillna("Missing"),
        Fare=fare["Fare"],
    )
    grouped_age = X[["Pclass", "Sex", "Age"]].skb.apply(GroupedAgeImputer())
    global_age = X[["Age"]].skb.apply(SimpleImputer(strategy="median"))

    grouped = common.assign(Age=grouped_age["Age"])
    baseline_features = add_interactions(grouped)
    global_features = add_interactions(common.assign(Age=global_age["Age"]))

    cat_base = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
    num_base = [
        "Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"
    ]
    cat_no_int = ["Pclass", "Sex", "Embarked", "Title"]
    num_no_int = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin"]

    cb_base, nn_base = make_probabilities(
        baseline_features, cat_base, num_base, y
    )
    cb_no_int, nn_no_int = make_probabilities(
        grouped, cat_no_int, num_no_int, y
    )
    cb_global, nn_global = make_probabilities(
        global_features, cat_base, num_base, y
    )

    # Exactly the four scored variants; the weighted variant reuses the
    # baseline prediction subgraphs. Thresholding uses >=, as in the original.
    variants = {
        "Baseline (50/50 Ensemble + Grouped Imputation + Interactions)":
            cb_base.skb.apply_func(
                blend_predictions, nn_base, 0.5, skrub.eval_mode()
            ),
        "Ablation 1 (Remove Interaction Features: Sex_Pclass, FamilyTier, TicketFreq)":
            cb_no_int.skb.apply_func(
                blend_predictions, nn_no_int, 0.5, skrub.eval_mode()
            ),
        "Ablation 2 (Global Median Age Imputation vs Grouped Median)":
            cb_global.skb.apply_func(
                blend_predictions, nn_global, 0.5, skrub.eval_mode()
            ),
        "Ablation 3 (Weighted Blend 0.85 CatBoost / 0.15 NN vs 50/50 Blend)":
            cb_base.skb.apply_func(
                blend_predictions, nn_base, 0.85, skrub.eval_mode()
            ),
    }
    pred = skrub.choose_from(variants, name="variant").as_data_op()

    # 4. Score — no cv= here; all variants use the marked outer splitter.
    # Reporting-only impact calculations and progress output are omitted.
    if __name__ == "__main__":
        np.random.seed(42)
        torch.manual_seed(42)
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
