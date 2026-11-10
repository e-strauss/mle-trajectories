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
FEATURE_COLS = CAT_COLS + NUM_COLS


class GroupAgeMedian(TransformerMixin, BaseEstimator):
    """Learn grouped Age medians; the global fallback is a separate node."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        medians = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X.assign(Age=X["Age"].fillna(medians))


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {"TicketFreq": X["Ticket"].map(self.counts_).fillna(1)},
            index=X.index,
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an evaluation set from the current fold's training rows."""

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
        # GetXY already carved the rows. This only forwards them to CatBoost.
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
            [nn.Embedding(num_c, e_dim) for num_c, e_dim in zip(cat_dims, emb_dims)]
        )
        layers = []
        prev_dim = sum(emb_dims) + num_features
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
        # Two output classes are explicit in the original architecture.
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.network(torch.cat(embs + [x_num], dim=1))


class NeuralStackingFeature(TransformerMixin, BaseEstimator):
    """The torch training loop; encoding and scaling remain separate nodes.

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
        return (
            torch.tensor(
                X[CAT_COLS].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=self.device_,
            ),
            torch.tensor(
                X[NUM_COLS].to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=self.device_,
            ),
        )

    def fit(self, X, y, *, X_val, y_val, cat_dims):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = np.unique(y)

        cat_dims = [int(dim) for dim in cat_dims]
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

        t_cat, t_num = self._tensors(X)
        t_y = torch.tensor(np.asarray(y), dtype=torch.long, device=self.device_)
        v_cat, v_num = self._tensors(X_val)

        for epoch in range(self.num_epochs):
            self.model_.train()
            permutation = torch.randperm(t_cat.size(0))

            for i in range(0, t_cat.size(0), self.batch_size):
                batch = permutation[i : i + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model_(t_cat[batch], t_num[batch])
                loss = criterion(outputs, t_y[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(v_cat, v_num)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_val, val_preds))

        # Preserve the original's final epoch, without best-epoch restoration.
        self.model_.eval()
        return self

    def fit_transform(self, X, y=None, *, X_val, y_val, cat_dims):
        self.fit(X, y, X_val=X_val, y_val=y_val, cat_dims=cat_dims)
        return self.transform(X_val)

    def transform(self, X):
        t_cat, t_num = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            probabilities = torch.softmax(
                self.model_(t_cat, t_num), dim=1
            )[:, 1].cpu().numpy()
        return pd.DataFrame({"p_nn": np.asarray(probabilities).ravel()})


class OptimizedBlend(ClassifierMixin, BaseEstimator):
    """Learn the original weight/threshold search on an inner evaluation set."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        cb_probs = np.asarray(X["p_cb"]).ravel()
        nn_probs = np.asarray(X["p_nn"]).ravel()
        labels = np.asarray(y).ravel()
        best_score = -1.0
        self.weight_ = 0.5
        self.threshold_ = 0.5

        # Preserve the original search order and strict ">" tie handling.
        for weight in np.linspace(0.0, 1.0, 101):
            blended = weight * cb_probs + (1.0 - weight) * nn_probs
            for threshold in np.linspace(0.1, 0.9, 81):
                score = accuracy_score(labels, (blended >= threshold).astype(int))
                if score > best_score:
                    best_score = score
                    self.weight_ = weight
                    self.threshold_ = threshold
        return self

    def predict(self, X):
        cb_probs = np.asarray(X["p_cb"]).ravel()
        nn_probs = np.asarray(X["p_nn"]).ravel()
        blended = self.weight_ * cb_probs + (1.0 - self.weight_) * nn_probs
        return (blended >= self.threshold_).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records the read without executing it.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the RAW target and reproduce the stratified hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and models. Medians and ticket frequencies were
    #    originally learned on the full table; separate estimators now learn
    #    these statistics from each outer fold's training rows only.
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

    grouped_age = X[["Pclass", "Sex", "Age"]].skb.apply(GroupAgeMedian())
    age = grouped_age[["Age"]].skb.apply(SimpleImputer(strategy="median"))
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))
    ticket_freq = X[["Ticket"]].skb.apply(TicketFrequency(), no_wrap=True)

    family_size = X["SibSp"] + X["Parch"] + 1
    family_tier = (
        family_size.astype(str)
        .where(family_size != 1, "IsAlone")
        .where(family_size == 1, "SmallFamily")
        .where(family_size <= 4, "LargeFamily")
    )

    features = X.assign(
        Title=title,
        Age=age["Age"],
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str),
        FamilySize=family_size,
        FamilyTier=family_tier,
        HasCabin=X["Cabin"].notna().astype(int),
        TicketFreq=ticket_freq["TicketFreq"],
        Embarked=X["Embarked"].fillna("Missing"),
        Fare=fare["Fare"],
    )[FEATURE_COLS]

    cb_features = features[CAT_COLS].astype(str).skb.concat(
        [features[NUM_COLS]], axis=1
    )

    # Sorted per-column vocabularies reproduce LabelEncoder. OrdinalEncoder
    # requires an unknown sentinel outside its learned codes: encode unknowns
    # as -1, then record a clip mapping just those values to zero, exactly
    # reproducing the original mapping.get(category, 0).
    nn_categories_encoded = features[CAT_COLS].astype(str).skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    )
    nn_categories = nn_categories_encoded.clip(lower=0)

    nn_numbers = features[NUM_COLS].astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_categories.skb.concat([nn_numbers], axis=1)

    # This statistic is needed only during model fitting, so pass it as a DataOp.
    cat_dims = (nn_categories.max(axis=0) + 1).to_numpy()

    # The original used its scored hold-out for CatBoost early stopping, the NN
    # scheduler, and blend optimization. Keep those procedures, but use an INNER
    # 20% stratified split of outer-training rows; the original optimistic score
    # cannot be reproduced without leakage. Both branches split identical rows.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    nn_parts = nn_features.skb.apply(GetXY(), y=y, no_wrap=True)

    X_cb_fit = cb_parts["X"]
    X_cb_val = cb_parts["X_val"]
    y_fit = cb_parts.get("y", y)
    y_val = cb_parts["y_val"]
    X_nn_fit = nn_parts["X"]
    X_nn_val = nn_parts["X_val"]

    # y_fit is a row subset of the unchanged raw labels, NOT a target transform.
    # Both GetXY nodes use the same stratification, fraction, and random seed,
    # so their validation rows stay aligned. No inverse needed.
    #
    # Each model node returns inner-validation probabilities from fit_transform
    # and holdout probabilities from transform.
    cb_probs = X_cb_fit.skb.apply(
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
            "eval_set": (X_cb_val, y_val),
            "early_stopping_rounds": 50,
        },
    )
    nn_probs = X_nn_fit.skb.apply(
        NeuralStackingFeature(),
        no_wrap=True,
        y=y_fit,
        fit_transform_kwargs={
            "X_val": X_nn_val,
            "y_val": y_val,
            "cat_dims": cat_dims,
        },
    )
    pred = cb_probs.skb.concat([nn_probs], axis=1).skb.apply(
        OptimizedBlend(), y=y_val
    )

    # 4. Score the single ensemble procedure. The marked splitter drives CV.
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
