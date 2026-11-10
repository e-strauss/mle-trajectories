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
    """Learn only the Pclass/Sex age medians."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        medians = pd.Series(self.medians_.reindex(keys).to_numpy(), index=X.index)
        return X.assign(Age=X["Age"].fillna(medians))


class TicketFrequency(TransformerMixin, BaseEstimator):
    """Learn the ticket-frequency lookup from the training rows."""

    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X.assign(TicketFreq=X["Ticket"].map(self.counts_).fillna(1))


class GetXY(TransformerMixin, BaseEstimator):
    """Expose a fold-local fit/evaluation split as recorded dictionary pieces."""

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


class TabularResBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(out_features, out_features)
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.bn1(x)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.linear1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)
        out = self.linear2(out)
        return residual + out


class TabularNeuralNet(nn.Module):
    def __init__(
        self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32), dropout=0.2
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, d) for n, d in zip(cat_dims, emb_dims)]
        )
        in_dim = sum(emb_dims) + num_features
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SiLU(),
        )

        blocks = []
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims:
            blocks.append(TabularResBlock(prev_dim, hidden_dim, dropout=dropout))
            prev_dim = hidden_dim
        self.res_blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.BatchNorm1d(prev_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(prev_dim, 2),
        )

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)
        ]
        x = torch.cat(embeddings + [x_num], dim=1) if embeddings else x_num
        return self.head(self.res_blocks(self.input_layer(x)))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """The original torch training loop, with preprocessing outside the model."""

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
        return (
            torch.tensor(
                X[CAT_COLS].to_numpy(),
                dtype=torch.long,
                device=self.device_,
            ),
            torch.tensor(
                X[NUM_COLS].to_numpy(),
                dtype=torch.float32,
                device=self.device_,
            ),
        )

    def fit(self, X, y, X_val=None, y_val=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cat_dims = np.asarray(cat_dims, dtype=int).tolist()
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
            optimizer, mode="max", factor=0.5, patience=10
        )

        train_cat, train_num = self._tensors(X)
        val_cat, val_num = self._tensors(X_val)
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )

        # Keep all 150 epochs: the original scheduled the learning rate but did
        # not restore a best checkpoint or stop the neural network early.
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
                val_outputs = self.model_(val_cat, val_num)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_preds)
            scheduler.step(val_acc)

        return self

    def predict(self, X):
        self.model_.eval()
        x_cat, x_num = self._tensors(X)
        with torch.no_grad():
            outputs = self.model_(x_cat, x_num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def ensemble_predictions(cb_prob, nn_prob, mode):
    # Predictor nodes contain fitted estimators, not probabilities, during fit.
    if mode == "fit":
        return None
    # prediction_type="Probability" is (n_samples, n_classes); column 1 is Survived.
    cb_prob = np.asarray(cb_prob, dtype=float)[:, 1]
    return (0.5 * cb_prob + 0.5 * nn_prob >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the read; importing this file performs no I/O.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data -- preserve the original stratified 80/20 hold-out.
    #    Mark the unmodified target.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Training-derived statistics are separate
    #    estimators, fixing the original full-table imputation/frequency leakage.
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
    features = X.assign(Title=title)
    features = features.skb.apply(GroupAgeImputer())

    # The fallback median is learned after the grouped age fill, as originally.
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Age"]
    )
    features = features.assign(
        Sex_Pclass=features["Sex"].astype(str)
        + "_"
        + features["Pclass"].astype(str)
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    features = features.assign(
        FamilyTier=features["FamilySize"]
        .le(4)
        .map({True: "SmallFamily", False: "LargeFamily"})
        .where(features["FamilySize"].ne(1), "IsAlone")
    )
    features = features.assign(HasCabin=features["Cabin"].notna().astype(int))
    features = features.skb.apply(TicketFrequency())
    features = features.assign(Embarked=features["Embarked"].fillna("Missing"))
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Fare"]
    )
    features = features[CAT_COLS + NUM_COLS]

    categorical_strings = features[CAT_COLS].astype(str)

    # CatBoost receives the same ordered string-categorical/numeric frame.
    cb_features = categorical_strings.skb.concat(
        [features[NUM_COLS]], axis=1
    )[CAT_COLS + NUM_COLS]

    # OrdinalEncoder sorts categories like LabelEncoder. Its unknown sentinel
    # must NOT collide with a learned code, so encode unknowns as -1 first,
    # then explicitly map them to 0 to reproduce the original fallback.
    nn_categories = categorical_strings.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    ).replace(-1, 0)
    nn_numbers = (
        features[NUM_COLS]
        .astype(np.float32)
        .skb.apply(StandardScaler())
    )
    nn_features = nn_categories.skb.concat(
        [nn_numbers], axis=1
    )[CAT_COLS + NUM_COLS]

    # Embedding cardinalities are fit-only statistics of the outer training
    # rows, supplied to the predictor.
    cat_dims = nn_categories.nunique().to_numpy()

    # 4. Fold-local evaluation sets.
    #    Both the original CatBoost early stopping and NN scheduler consulted
    #    the scored hold-out. Instead, reserve a stratified 20% of the fold's
    #    training rows, retaining the original split fraction and seed. This
    #    removes that leakage, so the original leaky score is not reproduced.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)

    cb_prob = cb_parts["X"].skb.apply(
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
    nn_prob = nn_parts["X"].skb.apply(
        NeuralPositiveProbability(),
        y=nn_parts.get("y", y),
        fit_kwargs={
            "X_val": nn_parts["X_val"],
            "y_val": nn_parts["y_val"],
            "cat_dims": cat_dims,
        },
    )

    # 5. Preserve the exact positive-probability average and >= 0.5 decoding.
    #    The original scores only this ensemble, not its individual components.
    pred = cb_prob.skb.apply_func(
        ensemble_predictions, nn_prob, skrub.eval_mode()
    )

    # 6. Score -- the splitter declared on mark_as_X drives validation.
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
