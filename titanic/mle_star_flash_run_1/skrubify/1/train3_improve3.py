import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = [
    "Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"
]


class GroupedAgeImputer(TransformerMixin, BaseEstimator):
    """Learn group medians and the original post-group-fill fallback median."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        filled = self._group_fill(X)
        self.fallback_ = filled.median()
        return self

    def _group_fill(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        medians = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X["Age"].fillna(medians)

    def transform(self, X):
        return self._group_fill(X).fillna(self.fallback_).to_frame("Age")


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).fillna(1).to_frame("TicketFreq")


class GetXY(TransformerMixin, BaseEstimator):
    """An eval set drawn only from the outer fold's training rows."""

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
        # Scored rows occupy "X". Keep "X_val" a real frame so a downstream
        # lookup still resolves when this node is evaluated at score time.
        return {"X": X, "X_val": X, "y": None, "y_val": None}


class CatBoostStackingFeature(CatBoostClassifier):
    """Inner-validation probabilities on fit, holdout probabilities on score."""

    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))

    def fit_transform(self, X, y=None, *, eval_set, early_stopping_rounds):
        self.fit(
            X,
            y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )
        return self.transform(eval_set[0])

    def transform(self, X):
        return pd.DataFrame({"p_cb": self.predict_proba(X)[:, 1]})


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


class NeuralStackingFeature(TransformerMixin, BaseEstimator):
    """The original torch training loop; preprocessing stays outside this node.

    fit_transform emits probabilities on the inner validation rows. transform
    emits them for the rows it is given, which at score time is the holdout.
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

    def _tensors(self, X):
        cats = torch.tensor(
            X[CAT_COLS].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=self.device_,
        )
        nums = torch.tensor(
            X[NUM_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        return cats, nums

    def fit(self, X, y, eval_set=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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

        x_cat, x_num = self._tensors(X)
        target = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        X_val, y_val = eval_set
        val_cat, val_num = self._tensors(X_val)

        for _ in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(x_cat.size(0))
            for start in range(0, x_cat.size(0), self.batch_size):
                batch = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(x_cat[batch], x_num[batch])
                loss = criterion(outputs, target[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                outputs = self.model_(val_cat, val_num)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_val, predictions))

        # As in the original, use the final epoch, not a best-epoch checkpoint.
        self.model_.eval()
        return self

    def fit_transform(self, X, y=None, *, eval_set, cat_dims):
        self.fit(X, y, eval_set=eval_set, cat_dims=cat_dims)
        return self.transform(eval_set[0])

    def transform(self, X):
        x_cat, x_num = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(x_cat, x_num)
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        return pd.DataFrame({"p_nn": np.asarray(probabilities).ravel()})


class OptimizedLogitEnsemble(ClassifierMixin, BaseEstimator):
    """Learn the weight/temperature/threshold search on an inner evaluation set."""

    def fit(self, X, y):
        eps = 1e-7
        cb_logits = logit(
            np.clip(np.asarray(X["p_cb"]).ravel(), eps, 1 - eps)
        )
        nn_logits = logit(
            np.clip(np.asarray(X["p_nn"]).ravel(), eps, 1 - eps)
        )
        labels = np.asarray(y).ravel()
        self.classes_ = np.unique(labels)
        self.weight_ = 0.5
        self.temperature_ = 1.0
        self.threshold_ = 0.5
        best_score = -1.0

        # Same grid, quantile thresholds, and strict ">" tie break as the script.
        for weight in np.linspace(0.0, 1.0, 101):
            blended = weight * cb_logits + (1.0 - weight) * nn_logits
            for temperature in np.logspace(-0.5, 0.5, 21):
                scaled = expit(blended / temperature)
                thresholds = np.unique(
                    np.quantile(scaled, np.linspace(0.01, 0.99, 100))
                )
                predictions = (scaled[:, None] >= thresholds[None, :]).astype(int)
                accuracies = (predictions == labels[:, None]).mean(axis=0)
                max_index = np.argmax(accuracies)
                if accuracies[max_index] > best_score:
                    best_score = accuracies[max_index]
                    self.weight_ = float(weight)
                    self.temperature_ = float(temperature)
                    self.threshold_ = float(thresholds[max_index])
        return self

    def predict(self, X):
        eps = 1e-7
        cb_logits = logit(
            np.clip(np.asarray(X["p_cb"]).ravel(), eps, 1 - eps)
        )
        nn_logits = logit(
            np.clip(np.asarray(X["p_nn"]).ravel(), eps, 1 - eps)
        )
        blended = (
            self.weight_ * cb_logits + (1.0 - self.weight_) * nn_logits
        )
        scaled = expit(blended / self.temperature_)
        return (scaled >= self.threshold_).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — the read is recorded; importing this file reads nothing.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and raw features. This is the exact stratified
    #    80/20 hold-out used by the original train_test_split.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Learned age/fare medians and ticket
    #    frequencies now use only outer-training rows, fixing the original's
    #    full-table preprocessing leakage.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = raw_title.replace(
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
    title = title.where(title.isin(["Mr", "Miss", "Mrs", "Master"]), "Rare")
    title = title.where(raw_title.notna(), "Missing")

    age = X[["Pclass", "Sex", "Age"]].skb.apply(GroupedAgeImputer())
    ticket_frequency = X[["Ticket"]].skb.apply(TicketFrequency())
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))

    family_size = X["SibSp"] + X["Parch"] + 1
    family_tier = (
        family_size.astype(str)
        .where(family_size == 1, "SmallFamily")
        .where(family_size <= 4, "LargeFamily")
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
    )[CAT_COLS + NUM_COLS]

    categories = features[CAT_COLS].astype(str)
    cb_features = categories.skb.concat([features[NUM_COLS]], axis=1)

    # OrdinalEncoder has the same sorted string codes as the per-column
    # LabelEncoders; unseen validation categories map to zero, as those encoders do.
    nn_categories = categories.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    ).replace(-1, 0).astype(np.int64)
    nn_numbers = features[NUM_COLS].astype(np.float32).skb.apply(
        StandardScaler()
    )
    nn_features = nn_categories.skb.concat([nn_numbers], axis=1)

    # 4. The original used its scored hold-out for CatBoost early stopping, the
    #    torch scheduler, and the weight/temperature/threshold search. Keep
    #    those procedures, but carve one stratified 20% eval set from the
    #    outer-training rows. Both branches use the same fraction and seed, so
    #    their validation rows stay aligned.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)
    y_fit = cb_parts.get("y", y)
    y_val = cb_parts["y_val"]

    # Each model node returns inner-validation probabilities from fit_transform
    # and holdout probabilities from transform. The ensemble freezes the search
    # on the inner labels, then scores the holdout with that frozen choice.
    cb_probs = cb_parts["X"].skb.apply(
        CatBoostStackingFeature(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            cat_features=CAT_COLS,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        no_wrap=True,
        y=y_fit,
        fit_transform_kwargs={
            "eval_set": (cb_parts["X_val"], y_val),
            "early_stopping_rounds": 50,
        },
    )
    nn_probs = nn_parts["X"].skb.apply(
        NeuralStackingFeature(),
        no_wrap=True,
        y=y_fit,
        fit_transform_kwargs={
            "eval_set": (nn_parts["X_val"], nn_parts["y_val"]),
            "cat_dims": nn_categories.nunique().tolist(),
        },
    )
    pred = cb_probs.skb.concat([nn_probs], axis=1).skb.apply(
        OptimizedLogitEnsemble(), y=y_val
    )

    # 5. Score the frozen ensemble. The marked splitter drives the hold-out.
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
