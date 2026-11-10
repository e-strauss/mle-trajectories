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
    """Learn grouped medians and the subsequent global fallback."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        self.fallback_ = self.transform_grouped(X).median()
        return self

    def transform_grouped(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        medians = pd.Series(self.medians_.reindex(keys).to_numpy(), index=X.index)
        # groupby.transform drops rows whose group key is null, even when Age
        # itself was present. The fallback median is taken after that wipe.
        valid_group = X["Pclass"].notna() & X["Sex"].notna()
        return X["Age"].fillna(medians).where(valid_group)

    def transform(self, X):
        return self.transform_grouped(X).fillna(self.fallback_).to_frame("Age")


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).fillna(1).to_frame("TicketFreq")


class GetXY(TransformerMixin, BaseEstimator):
    """Provide a training-only holdout for validation-driven training."""

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
            embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class TorchPositiveProbability(ClassifierMixin, BaseEstimator):
    """Only the torch training loop lives here; preprocessing stays in the plan."""

    def __init__(self, random_state=42, batch_size=32, num_epochs=150):
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_epochs = num_epochs

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
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dimensions = np.asarray(cat_dims, dtype=int).tolist()
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

        train_cat, train_num = self._tensors(X)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        X_val, y_val = eval_set
        val_cat, val_num = self._tensors(X_val)

        # Keep all 150 epochs and the validation-driven scheduler. The original
        # did not restore a best checkpoint or stop the neural net early.
        for _ in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for start in range(0, train_cat.size(0), self.batch_size):
                batch = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(train_cat[batch], train_num[batch])
                loss = criterion(outputs, train_y[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(val_cat, val_num)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_val, predictions))
        return self

    def predict(self, X):
        self.model_.eval()
        categorical, numeric = self._tensors(X)
        with torch.no_grad():
            outputs = self.model_(categorical, numeric)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def ensemble_predictions(cb_probabilities, nn_probabilities, mode):
    # prediction_type="Probability" returns (n_samples, n_classes) in classes_
    # order. classes_ is [0, 1], so column 1 is the positive-class probability
    # the net's predict already returns.
    if mode == "fit":
        return None
    cb_probabilities = np.asarray(cb_probabilities, dtype=float)[:, 1]
    probabilities = 0.5 * cb_probabilities + 0.5 * nn_probabilities
    return (probabilities >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records, but does not execute, the CSV read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and reproduce the original single
    #    stratified 80/20 holdout exactly.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Learned medians and ticket frequencies
    #    now use only outer-training rows, fixing the original full-table leaks.
    age = X[["Pclass", "Sex", "Age"]].skb.apply(
        GroupAgeImputer(), no_wrap=True
    )
    ticket_frequency = X[["Ticket"]].skb.apply(
        TicketFrequency(), no_wrap=True
    )

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
    ).fillna("Rare")
    title = title.where(raw_title.notna(), "Missing")

    features = X.assign(
        Title=title,
        Age=age["Age"],
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str),
        FamilySize=X["SibSp"] + X["Parch"] + 1,
    )
    family_size = features["FamilySize"]
    family_tier = family_size.astype(str).where(
        family_size <= 4, "LargeFamily"
    )
    family_tier = family_tier.where(~(family_size <= 4), "SmallFamily")
    family_tier = family_tier.where(family_size != 1, "IsAlone")
    features = features.assign(
        FamilyTier=family_tier,
        HasCabin=X["Cabin"].notna().astype(int),
        TicketFreq=ticket_frequency["TicketFreq"],
        Embarked=X["Embarked"].fillna("Missing"),
    )
    fare = features[["Fare"]].skb.apply(SimpleImputer(strategy="median"))
    features = features.assign(Fare=fare["Fare"])

    # Preserve the original explicit feature order and string representations.
    categorical_strings = features[CAT_COLS].astype(str)
    cb_features = categorical_strings.skb.concat([features[NUM_COLS]], axis=1)

    # OrdinalEncoder forbids an unknown_value that overlaps a known code.
    # Encode unknowns as -1, then replace that sentinel with zero as in the
    # original mapping. Known codes retain LabelEncoder's sorted 0-based order.
    nn_categories = categorical_strings.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    )
    nn_categories = nn_categories.replace(-1, 0)
    nn_numeric = features[NUM_COLS].astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_categories.skb.concat([nn_numeric], axis=1)
    category_dimensions = nn_categories.max(axis=0) + 1

    # Both original models used the scored holdout to guide training. Retain
    # CatBoost's patience and the neural scheduler, but provide matching
    # stratified 20% holdouts from outer-training rows only, visibly in the graph.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)

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
    nn_probabilities = nn_parts["X"].skb.apply(
        TorchPositiveProbability(
            random_state=42, batch_size=32, num_epochs=150
        ),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "eval_set": (nn_parts["X_val"], nn_parts["y_val"]),
            "cat_dims": category_dimensions,
        },
    )

    # Keep the original equal-weight positive probabilities and >= 0.5 rule.
    # This is one ensemble, not separately scored model variants.
    pred = cb_probabilities.skb.apply_func(
        ensemble_predictions, nn_probabilities, skrub.eval_mode()
    )

    # 4. Score — only the splitter on mark_as_X controls outer validation.
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
