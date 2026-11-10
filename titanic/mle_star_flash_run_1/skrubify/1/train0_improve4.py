import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "CabinDeck",
    "TicketPrefix",
    "Sex_Pclass",
    "Child_Pclass",
]
NUM_COLS = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "GroupSize",
    "FarePerPerson",
    "LogFare",
    "LogFarePerPerson",
    "IsChild",
]


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    """Learn one conditional median imputation and its fallback median."""

    def __init__(self, column, group_cols):
        self.column = column
        self.group_cols = group_cols

    def fit(self, X, y=None):
        self.medians_ = X.groupby(self.group_cols)[self.column].median()
        self.fallback_ = self._fill_groups(X).median()
        return self

    def _fill_groups(self, X):
        if len(self.group_cols) == 1:
            keys = pd.Index(X[self.group_cols[0]])
        else:
            keys = pd.MultiIndex.from_frame(X[self.group_cols])
        medians = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        # groupby.transform drops rows whose group key is null and replaces
        # the column. The fallback median is the median of what remains.
        valid_group = X[self.group_cols].notna().all(axis=1)
        return X[self.column].fillna(medians).where(valid_group)

    def transform(self, X):
        return X.assign(
            **{self.column: self._fill_groups(X).fillna(self.fallback_)}
        )


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        # The original maps value_counts and does not fill. A ticket absent
        # from the training fold has no count: an unseen non-null ticket is 1,
        # and a null ticket stays null.
        ticket = X["Ticket"]
        counts = ticket.map(self.counts_).fillna(1).where(ticket.notna())
        return X.assign(TicketCount=counts)


class GetXY(TransformerMixin, BaseEstimator):
    """Split this outer fold's training rows for model validation."""

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
            [nn.Embedding(n, e) for n, e in zip(cat_dims, emb_dims)]
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
    """The original torch loop, operating on prepared features only."""

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

    def fit(self, X, y, *, validation_data, cat_dims):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dimensions = [int(d) for d in cat_dims]
        embedding_dimensions = [
            max(2, int(round(d**0.5 * 2))) for d in dimensions
        ]
        # Preserve the original binary output width of 2.
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

        X_val, y_val = validation_data
        train_cat, train_num = self._tensors(X)
        val_cat, val_num = self._tensors(X_val)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )

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
                predictions = (
                    self.model_(val_cat, val_num)
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )
            scheduler.step(accuracy_score(y_val, predictions))

        # The original uses the final epoch, not a best-epoch checkpoint.
        return self

    def predict(self, X):
        self.model_.eval()
        categorical, numeric = self._tensors(X)
        with torch.no_grad():
            return (
                torch.softmax(self.model_(categorical, numeric), dim=1)[:, 1]
                .cpu()
                .numpy()
            )


def ensemble_predictions(cb_probability, nn_probability, mode):
    # prediction_type="Probability" returns (n_samples, n_classes) in classes_
    # order. classes_ is [0, 1], so column 1 is the positive-class probability
    # the net's predict already returns.
    if mode == "fit":
        return None
    cb_probability = np.asarray(cb_probability, dtype=float)[:, 1]
    return (
        0.5 * cb_probability + 0.5 * nn_probability >= 0.5
    ).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records, but does not execute, the CSV read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target. This splitter reproduces the
    #    original stratified train_test_split's exact outer hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and separate learned preprocessing nodes.
    title = (
        X["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
        .fillna("Missing")
        .replace(
            {
                "Mme": "Mrs",
                "Mlle": "Miss",
                "Ms": "Miss",
                "Lady": "Royalty",
                "Countess": "Royalty",
                "Dona": "Royalty",
                "Sir": "Royalty",
                "Jonkheer": "Royalty",
                "Don": "Royalty",
                "Capt": "Officer",
                "Col": "Officer",
                "Major": "Officer",
                "Dr": "Officer",
                "Rev": "Officer",
            }
        )
    )
    features = X.assign(Title=title)

    # The original learned medians before splitting. Learn them on each
    # outer training fold instead, fixing that preprocessing leakage.
    features = features.skb.apply(
        GroupMedianImputer("Age", ["Pclass", "Title"]), no_wrap=True
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )

    ticket_text = (
        features["Ticket"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.strip()
    )
    first_token = ticket_text.str.extract(r"^(\S+)", expand=False)
    multiple_tokens = ticket_text.str.contains(r"^\S+\s+\S+", regex=True)
    ticket_prefix = first_token.str.upper().where(
        multiple_tokens | ~first_token.str.isdigit(), "NUMERIC"
    )
    features = features.assign(TicketPrefix=ticket_prefix)

    # Ticket counts originally included validation rows. Learn frequencies
    # per fold; an unseen ticket receives singleton frequency.
    features = features.skb.apply(TicketFrequency(), no_wrap=True)
    features = features.assign(
        GroupSize=features["FamilySize"].skb.apply_func(
            np.maximum, features["TicketCount"]
        )
    )

    features = features.skb.apply(
        GroupMedianImputer("Fare", ["Pclass"]), no_wrap=True
    )
    features = features.assign(
        FarePerPerson=features["Fare"] / features["GroupSize"]
    )
    features = features.assign(
        LogFare=features["Fare"].skb.apply_func(np.log1p)
    )
    features = features.assign(
        LogFarePerPerson=features["FarePerPerson"].skb.apply_func(np.log1p)
    )

    raw_deck = features["Cabin"].str[0].fillna("Missing")
    features = features.assign(
        CabinDeck=raw_deck.map(
            {
                "A": "ABC",
                "B": "ABC",
                "C": "ABC",
                "D": "DE",
                "E": "DE",
                "F": "FG",
                "G": "FG",
                "T": "ABC",
                "Missing": "Missing",
            }
        ).fillna("Missing")
    )
    features = features.assign(
        Sex_Pclass=features["Sex"].astype(str)
        + "_"
        + features["Pclass"].astype(str)
    )
    features = features.assign(IsChild=(features["Age"] < 12).astype(int))
    features = features.assign(
        Child_Pclass=features["IsChild"].astype(str)
        + "_"
        + features["Pclass"].astype(str)
    )
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing")
    )

    categorical = features[CAT_COLS].astype(str)
    numeric = features[NUM_COLS]
    cb_features = categorical.skb.concat([numeric], axis=1)

    # The original early-stopped CatBoost and scheduled the NN on the scored
    # hold-out. Preserve those mechanisms but use validation rows carved from
    # the outer TRAINING fold. Identical inner splits keep the branches aligned.
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

    # Sorted string categories match LabelEncoder. OrdinalEncoder forbids
    # unknown_value=0 because 0 already encodes a known category. Use its
    # valid sentinel -1, then explicitly map it to the original fallback 0.
    nn_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        ).set_output(transform="pandas")
    )
    nn_categorical = nn_categorical.replace(-1, 0)

    nn_numeric = numeric.astype(np.float32).skb.apply(
        StandardScaler().set_output(transform="pandas")
    )
    nn_features = nn_categorical.skb.concat([nn_numeric], axis=1)[
        CAT_COLS + NUM_COLS
    ]

    # These cardinalities are fit-only inputs, computed visibly from the
    # outer training fold and passed through fit_kwargs.
    cat_dims = (nn_categorical.max() + 1).astype(int).tolist()
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
            "validation_data": (nn_parts["X_val"], nn_parts["y_val"]),
            "cat_dims": cat_dims,
        },
    )

    # Preserve probability averaging and the inclusive 0.5 threshold. Gate
    # arithmetic because fit-mode prediction nodes contain fitted estimators.
    pred = cb_probability.skb.apply_func(
        ensemble_predictions, nn_probability, skrub.eval_mode()
    )

    # 4. Score the original ensemble only; the splitter is on mark_as_X.
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
