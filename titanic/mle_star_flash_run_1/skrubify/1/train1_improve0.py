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

CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    """Learn grouped Age medians and the post-imputation fallback median."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        group_values = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        self.fallback_ = X["Age"].fillna(group_values).median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        group_values = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        return (
            X["Age"]
            .fillna(group_values)
            .fillna(self.fallback_)
            .to_frame("Age")
        )


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).fillna(1).to_frame("TicketFreq")


class GetXY(TransformerMixin, BaseEstimator):
    """Make CatBoost's per-fit early-stopping split visible in the plan."""

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
    """Original torch loop, including best-validation-loss checkpointing."""

    def __init__(
        self,
        num_epochs=150,
        batch_size=32,
        random_state=42,
        validation_fraction=0.2,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.validation_fraction = validation_fraction

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Encoding and scaling are separate recorded estimator nodes.
        x_cat = X[CAT_COLS].to_numpy(dtype=np.int64)
        x_num = X[NUM_COLS].to_numpy(dtype=np.float32)
        cat_dims = (x_cat.max(axis=0) + 1).tolist()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

        self.model_ = TabularNeuralNet(
            cat_dims, emb_dims, num_features=len(NUM_COLS)
        ).to(self.device_)

        # This is an INNER split for the torch loop's checkpoint selection,
        # not the outer validation split (which lives on mark_as_X).
        # The original selected checkpoints on the scored hold-out itself;
        # use only this fold's training rows to avoid that leakage.
        train_idx, val_idx = train_test_split(
            np.arange(len(y)),
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=y,
        )
        target = np.asarray(y)
        t_cat = torch.tensor(x_cat, dtype=torch.long, device=self.device_)
        t_num = torch.tensor(x_num, dtype=torch.float32, device=self.device_)
        t_y = torch.tensor(target, dtype=torch.long, device=self.device_)
        train_cat, val_cat = t_cat[train_idx], t_cat[val_idx]
        train_num, val_num = t_num[train_idx], t_num[val_idx]
        train_y, val_y = t_y[train_idx], t_y[val_idx]

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=1e-5
        )

        best_val_loss = float("inf")
        best_model_state = None
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

            scheduler.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(
                    self.model_(val_cat, val_num), val_y
                ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    key: value.cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }

        if best_model_state is not None:
            self.model_.load_state_dict(
                {
                    key: value.to(self.device_)
                    for key, value in best_model_state.items()
                }
            )
        self.model_.eval()
        return self

    def predict(self, X):
        x_cat = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        x_num = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(x_cat, x_num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def ensemble_predictions(cb_probs, nn_probs, mode):
    # prediction_type="Probability" is (n_samples, n_classes); column 1 is P(1).
    if mode == "fit":
        return None
    cb_probs = np.asarray(cb_probs, dtype=float)[:, 1]
    # Preserve the equal-weight arithmetic and inclusive threshold.
    return (0.5 * cb_probs + 0.5 * nn_probs >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read; importing does not read the CSV.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the RAW target and design matrix.
    # This reproduces the original stratified train_test_split's exact rows.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and separate learned preprocessing.
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

    # The original computed these statistics on the whole table, leaking
    # validation information. Fit each statistic on training rows only.
    age = X[["Pclass", "Sex", "Age"]].skb.apply(
        GroupAgeImputer(), no_wrap=True
    )
    ticket_freq = X[["Ticket"]].skb.apply(
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
        family_size.map({1: "IsAlone"})
        .fillna("SmallFamily")
        .where(family_size <= 4, "LargeFamily")
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=X["Cabin"].notna().astype(int))
    features = features.assign(TicketFreq=ticket_freq["TicketFreq"])
    features = features.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.assign(Fare=fare["Fare"])
    features = features[CAT_COLS + NUM_COLS]

    categories = features[CAT_COLS].astype(str)
    numbers = features[NUM_COLS]
    cb_features = categories.skb.concat([numbers], axis=1)

    # 4. CatBoost -- same hyperparameters and 50-round early stopping.
    # The original eval set was the scored hold-out; instead carve an eval
    # set out of this fold's own training rows using recorded GetXY.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    cb_probs = cb_parts["X"].skb.apply(
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

    # 5. Neural-network preprocessing.
    # OrdinalEncoder sorts categories like LabelEncoder, but its unknown
    # sentinel cannot collide with a known code. Encode unknowns as -1,
    # then map just that sentinel to 0 with a recorded clip operation,
    # reproducing the original mapping.get(value, 0) behavior.
    nn_categories = categories.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    ).clip(lower=0)
    nn_numbers = numbers.astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_categories.skb.concat([nn_numbers], axis=1)
    nn_features = nn_features[CAT_COLS + NUM_COLS]
    nn_probs = nn_features.skb.apply(
        NeuralPositiveProbability(
            num_epochs=150,
            batch_size=32,
            random_state=42,
            validation_fraction=0.2,
        ),
        y=y,
    )

    # 6. Only the ensemble was scored in the original.
    # Gate arithmetic because fit-mode predictor nodes contain fitted models.
    pred = cb_probs.skb.apply_func(
        ensemble_predictions, nn_probs, skrub.eval_mode()
    )

    # 7. Score the original single hold-out using accuracy.
    # Removing preprocessing and model-selection leakage can change the score.
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
