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


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"]
NUM_COLS = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "TicketCount",
    "FarePerPerson",
    "LogFare",
    "LogFarePerPerson",
    "CabinCount",
    "Age_Pclass",
    "IsChild",
]
TITLE_MAPPING = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Royalty",
    "Countess": "Royalty",
    "Capt": "Officer",
    "Col": "Officer",
    "Don": "Royalty",
    "Dr": "Officer",
    "Major": "Officer",
    "Rev": "Officer",
    "Sir": "Royalty",
    "Jonkheer": "Royalty",
    "Dona": "Royalty",
}


class TicketFrequency(TransformerMixin, BaseEstimator):
    """Learn ticket frequencies from the current fold's training rows."""

    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {
                "TicketCount": X["Ticket"]
                .map(self.counts_)
                .fillna(1)
                .astype(int)
            },
            index=X.index,
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Keep validation-dependent training inside the outer training fold."""

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
    """Only the torch model and its per-fit training loop live in this node."""

    def __init__(self, batch_size=32, num_epochs=150, random_state=42):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
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

    def fit(self, X, y, eval_set=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        dimensions = np.asarray(cat_dims, dtype=int).tolist()
        embedding_dimensions = [
            max(2, int(round(dimension**0.5 * 2)))
            for dimension in dimensions
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

        X_val, y_val = eval_set
        train_cat, train_num = self._tensors(X)
        val_cat, val_num = self._tensors(X_val)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )

        # Preserve all 150 epochs, final-epoch predictions, and the original
        # validation-accuracy scheduler; there was no best-epoch restoration.
        for epoch in range(self.num_epochs):
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
                predictions = (
                    self.model_(val_cat, val_num)
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )
            scheduler.step(accuracy_score(y_val, predictions))

        self.model_.eval()
        return self

    def predict_proba(self, X):
        categorical, numerical = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            return (
                torch.softmax(self.model_(categorical, numerical), dim=1)
                .cpu()
                .numpy()
            )

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def ensemble_predictions(cb_probability, nn_probability, mode):
    # Predictor nodes contain estimators rather than probabilities in fit mode.
    # prediction_type="Probability" returns (n_samples, n_classes) in classes_
    # order. classes_ is [0, 1], so column 1 is the positive-class probability
    # the net's predict already returns.
    if mode == "fit":
        return None
    cb_probability = np.asarray(cb_probability, dtype=float)[:, 1]
    return (0.5 * cb_probability + 0.5 * nn_probability >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this file records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target. This splitter reproduces the
    #    original single stratified 80/20 train_test_split exactly.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Median imputation and ticket frequencies
    #    originally used the entire table; learn them on outer training rows
    #    instead, removing that leakage.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    medians = X[["Age", "Fare"]].skb.apply(
        SimpleImputer(strategy="median", keep_empty_features=True)
    )
    ticket_count = X[["Ticket"]].skb.apply(TicketFrequency())
    deck = X["Cabin"].str[0].fillna("Missing")
    deck = deck.where(
        deck.isin(["A", "B", "C", "D", "E", "F", "G", "T"]), "Missing"
    )

    features = X.assign(
        Title=raw_title.map(TITLE_MAPPING).fillna("Other"),
        Age=medians["Age"],
        Fare=medians["Fare"],
        Embarked=X["Embarked"].fillna("Missing"),
        CabinDeck=deck,
        # Counting non-whitespace runs matches the original len(cabin.split()),
        # including empty or whitespace-only cabin strings.
        CabinCount=X["Cabin"].fillna("").str.count(r"\S+"),
        FamilySize=X["SibSp"] + X["Parch"] + 1,
        TicketCount=ticket_count["TicketCount"],
    )
    group_size = features["FamilySize"].skb.apply_func(
        np.maximum, features["TicketCount"]
    )
    features = features.assign(
        FarePerPerson=features["Fare"] / group_size,
        LogFare=features["Fare"].skb.apply_func(np.log1p),
        Age_Pclass=features["Age"] * features["Pclass"],
        IsChild=(features["Age"] < 12).astype(int),
    )
    features = features.assign(
        LogFarePerPerson=features["FarePerPerson"].skb.apply_func(np.log1p)
    )
    categorical = features[CAT_COLS].astype(str)
    numerical = features[NUM_COLS]
    cb_features = categorical.skb.concat([numerical], axis=1)

    # 4. CatBoost — retain every model parameter and 50-round patience.
    #    The original early-stopped on its scored rows. An inner stratified
    #    20% eval set now comes only from outer training rows; consequently
    #    that leaky original score cannot be reproduced exactly.
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

    # 5. Neural preprocessing — separate fitted nodes for categorical encoding
    #    and scaling. Sorted ordinal codes reproduce LabelEncoder per column.
    #    OrdinalEncoder forbids an unknown sentinel that overlaps a known code:
    #    encode unknowns as -1, then map them to 0 with a recorded clip,
    #    preserving the original mapping.get(value, 0) semantics.
    nn_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    ).clip(lower=0)
    nn_numerical = numerical.astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_categorical.skb.concat([nn_numerical], axis=1)

    # Embedding sizes are fit-only statistics, exposed as recorded operations.
    cat_dims = nn_categorical.max(axis=0) + 1

    # The original scheduler also inspected scored validation labels. Give it
    # the same inner row split as CatBoost, never the outer scored rows.
    # Encoders/scaler remain fitted only on outer training rows.
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_probability = nn_parts["X"].skb.apply(
        NeuralPositiveProbability(
            batch_size=32, num_epochs=150, random_state=42
        ),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "eval_set": (nn_parts["X_val"], nn_parts["y_val"]),
            "cat_dims": cat_dims,
        },
    )

    # 6. Preserve the single scored ensemble, equal probability weights, and
    #    inclusive 0.5 threshold. Individual models were not scored variants.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 7. Score — the splitter lives exclusively on mark_as_X.
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
