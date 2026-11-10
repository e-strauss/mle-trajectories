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


class TicketFrequency(TransformerMixin, BaseEstimator):
    """Learn the ticket-frequency lookup on the current training fold."""

    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        # A previously unseen, nonmissing ticket denotes a singleton group.
        # Preserve missing-ticket values rather than imputing them.
        frequency = (
            X["Ticket"].map(self.counts_).fillna(1).where(X["Ticket"].notna())
        )
        return pd.DataFrame({"TicketFreq": frequency}, index=X.index)


class GetXY(TransformerMixin, BaseEstimator):
    """Reserve early-stopping rows from this fold's training rows only."""

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
            [
                nn.Embedding(num_c, e_dim)
                for num_c, e_dim in zip(cat_dims, emb_dims)
            ]
        )
        in_dim = sum(emb_dims) + num_features
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.network(torch.cat(embs + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """The original torch training loop, with a private scheduler hold-out."""

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

    def fit(self, X, y, eval_set, cat_dims):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Encoding and scaling are separate graph nodes. The scheduler
        # hold-out arrives from GetXY; X here is that node's training rows.
        dims = np.asarray(cat_dims).ravel().astype(int).tolist()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in dims]
        X_train, y_train = X, y
        X_val, y_val = eval_set
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

        t_cat_train = torch.tensor(
            X_train[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        t_num_train = torch.tensor(
            X_train[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        t_y_train = torch.tensor(
            np.asarray(y_train), dtype=torch.long, device=self.device_
        )
        t_cat_val = torch.tensor(
            X_val[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        t_num_val = torch.tensor(
            X_val[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )

        # Keep all 150 epochs and the final epoch's weights: the original
        # did not restore a best checkpoint or stop its torch loop early.
        for epoch in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(t_cat_train.size(0))
            for start in range(0, t_cat_train.size(0), self.batch_size):
                batch = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(t_cat_train[batch], t_num_train[batch])
                loss = criterion(outputs, t_y_train[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(t_cat_val, t_num_val)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, predictions)
            scheduler.step(val_acc)

        self.model_.eval()
        return self

    def predict(self, X):
        t_cat = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        t_num = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(t_cat, t_num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def ensemble_predictions(cb_probability, nn_probability, mode):
    # prediction_type="Probability" returns (n_samples, n_classes) in classes_
    # order. classes_ is [0, 1], so column 1 is the positive-class probability
    # the net's predict already returns.
    if mode == "fit":
        return None
    cb_probability = np.asarray(cb_probability, dtype=float)[:, 1]
    probabilities = 0.5 * cb_probability + 0.5 * nn_probability
    return (probabilities >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this module records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and features. This reproduces the original's
    #    stratified 80/20 outer train_test_split, including its random seed.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    raw_title = X["Name"].str.extract(
        r" ([A-Za-z]+)\.", expand=False
    ).fillna("Missing")
    features = X.assign(Title=raw_title.map(TITLE_MAP).fillna("Rare"))

    # The original learned medians and ticket frequencies before splitting.
    # Learn each on the training fold instead, removing that leakage.
    imputed = features[["Age", "Fare"]].skb.apply(
        SimpleImputer(strategy="median")
    )
    ticket_frequency = X[["Ticket"]].skb.apply(TicketFrequency())
    features = features.assign(
        Age=imputed["Age"],
        Fare=imputed["Fare"],
        Embarked=features["Embarked"].fillna("Missing"),
        CabinDeck=features["Cabin"].str[0].fillna("Missing"),
        FamilySize=features["SibSp"] + features["Parch"] + 1,
        TicketFreq=ticket_frequency["TicketFreq"],
    )
    features = features.assign(
        GroupSize=features["FamilySize"].skb.apply_func(
            np.maximum, features["TicketFreq"]
        )
    )
    features = features.assign(
        FarePerPerson=features["Fare"] / features["GroupSize"]
    )
    features = features.assign(
        LogFarePerPerson=features["FarePerPerson"].skb.apply_func(np.log1p),
        IsChild=(features["Age"] < 12).astype(int),
        IsMother=(
            (features["Sex"] == "female")
            & (features["Parch"] > 0)
            & (features["Title"] == "Mrs")
        ).astype(int),
        Sex_Pclass=features["Sex"].astype(str)
        + "_"
        + features["Pclass"].astype(str),
    )
    features = features[FEATURE_COLS]

    cb_categories = features[CAT_COLS].astype(str)
    cb_features = cb_categories.skb.concat(
        [features[NUM_COLS]], axis=1
    )[FEATURE_COLS]

    # The original early-stopped CatBoost on the scored outer validation rows.
    # Keep its patience and iterations, but reserve an inner 20% of the fold's
    # training rows: reproducing the original leaky score is not possible.
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

    # OrdinalEncoder uses the same sorted string categories as the original
    # per-column LabelEncoders. Map unseen categories back to index zero.
    nn_categories = features[CAT_COLS].astype(str).skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    )
    nn_categories = nn_categories.clip(lower=0).astype(np.int64)
    nn_numeric = features[NUM_COLS].astype(np.float32).skb.apply(
        StandardScaler()
    )
    nn_features = nn_categories.skb.concat(
        [nn_numeric], axis=1
    )[FEATURE_COLS]

    # The torch scheduler also used the scored validation rows originally.
    # The hold-out is a recorded GetXY node on this fold's training rows.
    # Embedding sizes still come from the whole outer training frame, which
    # is what the original encoded before it split.
    cat_dimensions = nn_features[CAT_COLS].max(axis=0) + 1
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_probability = nn_parts["X"].skb.apply(
        NeuralPositiveProbability(
            batch_size=32,
            num_epochs=150,
            learning_rate=1e-2,
            weight_decay=1e-4,
            random_state=42,
        ),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "eval_set": (nn_parts["X_val"], nn_parts["y_val"]),
            "cat_dims": cat_dimensions,
        },
    )

    # Preserve the manual 50/50 positive-probability blend and >= 0.5 threshold.
    # These are components of ONE scored ensemble, not separately scored variants.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 4. Score — the splitter lives exclusively on mark_as_X.
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
