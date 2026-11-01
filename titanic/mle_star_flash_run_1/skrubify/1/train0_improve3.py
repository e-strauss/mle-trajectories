import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck", "Sex_Pclass"]
NUM_COLS = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "GroupSize",
    "FarePerPerson",
    "LogFarePerPerson",
    "IsChild",
    "IsMother",
]
FEATURE_COLS = CAT_COLS + NUM_COLS

TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Ms": "Miss",
    "Capt": "Rare",
    "Lady": "Rare",
    "Jonkheer": "Rare",
    "Don": "Rare",
    "Dona": "Rare",
    "Countess": "Rare",
    "Sir": "Rare",
}


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_c, emb_dim)
                for num_c, emb_dim in zip(cat_dims, emb_dims)
            ]
        )
        layers = []
        previous = sum(emb_dims) + num_features
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


def as_model_frame(frame):
    """Feature columns in the original order, with categoricals as strings.

    Called on DataOps so the projection and each ``astype(str)`` stay in the plan.
    """
    selected = frame[FEATURE_COLS]
    return selected.assign(**{
        column: selected[column].astype(str) for column in CAT_COLS
    })


def _original_split(y):
    y = np.asarray(y).ravel()
    indices = np.arange(len(y))
    return train_test_split(
        indices, y, test_size=0.2, random_state=42, stratify=y
    )


def holdout_rows(df):
    _, idx_val, _, _ = _original_split(df["Survived"])
    return df.iloc[idx_val]


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Train the original network on the outer training rows.

    Categorical codes and the numeric scaler are fit on those rows only.
    The scheduler sees the scored hold-out, which is the original script's
    validation set. Unseen categories map to code 0.
    """

    def __init__(
        self,
        batch_size=32,
        num_epochs=150,
        learning_rate=1e-2,
        weight_decay=1e-4,
        random_state=42,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _cat_matrix(self, X):
        columns = []
        for column, encoder in zip(CAT_COLS, self.encoders_):
            mapping = {
                value: code for code, value in enumerate(encoder.classes_)
            }
            encoded = np.array(
                [
                    mapping.get(value, 0)
                    for value in X[column].astype(str).to_numpy()
                ]
            )
            columns.append(encoded)
        return np.stack(columns, axis=1).astype(np.int64)

    def _numeric_matrix(self, X):
        return self.scaler_.transform(
            X[NUM_COLS].to_numpy().astype(np.float32)
        )

    def _tensors(self, categorical, numeric):
        return (
            torch.tensor(categorical, dtype=torch.long, device=self.device_),
            torch.tensor(numeric, dtype=torch.float32, device=self.device_),
        )

    def fit(self, X, y, *, X_val, y_val):
        y = np.asarray(y).ravel()
        y_val = np.asarray(y_val).ravel()
        self.classes_ = np.unique(y)
        self.encoders_ = [
            LabelEncoder().fit(X[column].astype(str)) for column in CAT_COLS
        ]
        self.scaler_ = StandardScaler().fit(
            X[NUM_COLS].to_numpy().astype(np.float32)
        )
        train_cat = self._cat_matrix(X)
        val_cat = self._cat_matrix(X_val)
        train_num = self._numeric_matrix(X)
        val_num = self._numeric_matrix(X_val)

        # Encoding above is NumPy-only. Seed torch once the weights are next,
        # matching the original manual_seed before TabularNeuralNet().
        torch.manual_seed(self.random_state)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dimensions = [len(encoder.classes_) for encoder in self.encoders_]
        embedding_dimensions = [
            max(2, int(round(dimension**0.5 * 2)))
            for dimension in dimensions
        ]
        self.model_ = TabularNeuralNet(
            dimensions, embedding_dimensions, num_features=len(NUM_COLS)
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
        train_cat_t, train_num_t = self._tensors(train_cat, train_num)
        val_cat_t, val_num_t = self._tensors(val_cat, val_num)
        train_y = torch.tensor(y, dtype=torch.long, device=self.device_)

        # Retain all 150 epochs and the final weights, not a best checkpoint.
        for _epoch in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(train_cat_t.size(0))
            for start in range(0, train_cat_t.size(0), self.batch_size):
                batch_indices = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(
                    train_cat_t[batch_indices], train_num_t[batch_indices]
                )
                loss = criterion(outputs, train_y[batch_indices])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(val_cat_t, val_num_t)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_val, predictions))

        self.model_.eval()
        return self

    def predict(self, X):
        categorical, numeric = self._tensors(
            self._cat_matrix(X), self._numeric_matrix(X)
        )
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(categorical, numeric)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def _positive_class_probability(values):
    """Column 1 of a probability matrix, or a vector that is already that column.

    ``CatBoostClassifier.predict(prediction_type="Probability")`` returns shape
    ``(n_samples, n_classes)`` in ``classes_`` order. For this target that order
    is ``[0, 1]``. The network's ``predict`` already returns column 1.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        return array[:, 1]
    return array.reshape(-1)


class MeanProbabilityEnsemble(ClassifierMixin, BaseEstimator):
    """Threshold the equal-weight average of two positive-class probabilities.

    The members are the upstream classifier nodes. This estimator does not fit
    them: ``X`` is the CatBoost probability and ``nn_probability`` is the
    neural net's. Averaging those two vectors is the whole ``predict``.
    """

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
    # 1. Load Data — importing records the read without accessing the file.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Feature engineering on the full table. The original imputes Age and
    # Fare and counts tickets, including the rows it later scores, before it splits.
    raw_title = (
        data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Missing")
    )
    features = data.assign(Title=raw_title.map(TITLE_MAP).fillna("Rare"))
    features = features.assign(Age=features["Age"].fillna(features["Age"].median()))
    features = features.assign(Fare=features["Fare"].fillna(features["Fare"].median()))
    features = features.assign(Embarked=features["Embarked"].fillna("Missing"))
    features = features.assign(CabinDeck=features["Cabin"].str[0].fillna("Missing"))
    features = features.assign(FamilySize=features["SibSp"] + features["Parch"] + 1)
    features = features.assign(
        TicketFreq=features["Ticket"].map(features["Ticket"].value_counts())
    )
    features = features.assign(
        GroupSize=features["FamilySize"].skb.apply_func(
            np.maximum, features["TicketFreq"]
        )
    )
    features = features.assign(FarePerPerson=features["Fare"] / features["GroupSize"])
    features = features.assign(
        LogFarePerPerson=features["FarePerPerson"].skb.apply_func(np.log1p)
    )
    features = features.assign(IsChild=(features["Age"] < 12).astype(int))
    features = features.assign(
        IsMother=(
            (features["Sex"] == "female")
            & (features["Parch"] > 0)
            & (features["Title"] == "Mrs")
        ).astype(int)
    )
    features = features.assign(
        Sex_Pclass=features["Sex"].astype(str) + "_" + features["Pclass"].astype(str)
    )

    # The scored hold-out, computed on the full table so it stays available
    # inside fit for CatBoost early stopping and the network scheduler.
    holdout = features.skb.apply_func(holdout_rows)
    X_val = as_model_frame(holdout)
    y_val = holdout["Survived"]

    # 3. Same stratified hold-out as the original train_test_split.
    y = features["Survived"].skb.mark_as_y()
    X = features.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )
    X_model = as_model_frame(X)

    # 4. CatBoost early-stops on the scored hold-out, as the original does.
    # cat_features is a fit argument: the constructor form fails clone.
    # prediction_type asks for the probability matrix; column 1 is taken in
    # MeanProbabilityEnsemble.
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

    # 5. The network trains on every outer training row. Its scheduler and
    # final probabilities use the same scored hold-out.
    nn_probability = X_model.skb.apply(
        NeuralPositiveProbability(),
        y=y,
        fit_kwargs={"X_val": X_val, "y_val": y_val},
    )

    # 6. Preserve the equal-weight probability average and inclusive threshold.
    # The original scores only this ensemble, not its components separately.
    # VotingClassifier would drop cat_features, eval_set, and the net's
    # validation frame, so the average stays in this estimator.
    pred = cb_probability.skb.apply(
        MeanProbabilityEnsemble(),
        y=y,
        fit_kwargs={"nn_probability": nn_probability},
        predict_kwargs={"nn_probability": nn_probability},
    )

    # 7. Score the original outer hold-out, with no CV override.
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
