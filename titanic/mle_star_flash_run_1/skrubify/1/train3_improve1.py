import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, train_test_split
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


def original_split_metadata(target):
    """Preserve the source's fixed base/meta split and its shuffled row order.

    These columns are splitter metadata, never predictive features. Computing
    the stratified split from labels is the same operation as a CV splitter.
    """
    base, meta = train_test_split(
        np.arange(len(target)), test_size=0.2, random_state=42,
        stratify=np.asarray(target),
    )
    order = np.empty(len(target), dtype=np.int64)
    order[base] = np.arange(len(base))
    order[meta] = np.arange(len(meta))
    is_base = np.zeros(len(target), dtype=bool)
    is_base[base] = True
    return pd.DataFrame(
        {"_base_training": is_base, "_source_order": order}, index=target.index
    )


class MetaValidationCV(BaseCrossValidator):
    """Five meta folds; the original base-training rows are always in train."""

    def get_n_splits(self, X=None, y=None, groups=None):
        return 5

    def split(self, X, y=None, groups=None):
        base = np.flatnonzero(X["_base_training"].to_numpy())
        meta = np.flatnonzero(~X["_base_training"].to_numpy())
        order = X["_source_order"].to_numpy()
        base = base[np.argsort(order[base])]
        meta = meta[np.argsort(order[meta])]
        target = np.asarray(y)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train, validation in cv.split(np.zeros((len(meta), 1)), target[meta]):
            yield np.concatenate([base, meta[train]]), meta[validation]


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


class NeuralProbability(ClassifierMixin, BaseEstimator):
    """The original torch model and training loop, inside the ensemble."""

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

    def predict(self, X):
        t_cat, t_num = self._tensors(X)
        self.model_.eval()
        with torch.no_grad():
            probabilities = torch.softmax(
                self.model_(t_cat, t_num), dim=1
            )[:, 1].cpu().numpy()
        return np.asarray(probabilities).ravel()


class LogitStackingClassifier(ClassifierMixin, BaseEstimator):
    """Fit the genuine CatBoost/torch/logistic ensemble for one meta fold.

    The fixed base-training rows and meta-training rows retain their original
    roles. Each of the five scored meta partitions gets its own logistic fit;
    none of its labels influence model fitting or early stopping.
    """

    def _neural_matrix(self, X):
        categories = self.encoder_.transform(X[CAT_COLS].astype(str))
        categories = np.maximum(categories, 0).astype(np.int64)
        numeric = self.scaler_.transform(
            X[NUM_COLS].to_numpy(dtype=np.float32)
        )
        return pd.DataFrame(
            np.column_stack([categories, numeric]), columns=FEATURE_COLS,
            index=X.index,
        )

    def _meta_features(self, X):
        cb = self.catboost_.predict_proba(X[FEATURE_COLS])[:, 1]
        neural = self.neural_.predict(self._neural_matrix(X))
        # Preserve separate clipping/logit evaluation before concatenation:
        # the source transforms NN probabilities in float32, CatBoost in float64.
        cb_logit = logit(np.clip(cb, 1e-6, 1 - 1e-6))
        neural_logit = logit(np.clip(neural, 1e-6, 1 - 1e-6))
        return np.hstack([cb_logit.reshape(-1, 1), neural_logit.reshape(-1, 1)])

    def fit(self, X, y):
        target = np.asarray(y)
        is_base = X["_base_training"].to_numpy(dtype=bool)
        base = X.loc[is_base, FEATURE_COLS]
        meta = X.loc[~is_base, FEATURE_COLS]
        base_target = target[is_base]
        meta_target = target[~is_base]
        self.classes_ = np.unique(target)

        # The original encoders/scaler fit only the fixed base 80%, not the
        # meta-fold training rows. Preserve this legitimate training scope.
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            dtype=np.int64,
        ).fit(base[CAT_COLS].astype(str))
        self.scaler_ = StandardScaler().fit(
            base[NUM_COLS].to_numpy(dtype=np.float32)
        )
        base_neural = self._neural_matrix(base)
        cat_dims = [len(categories) for categories in self.encoder_.categories_]

        # Fix scored-split early-stopping/scheduler leakage: reserve only base
        # training rows for eval, retaining the original patience and budgets.
        fit, evaluation = train_test_split(
            np.arange(len(base)), test_size=0.2, random_state=42,
            stratify=base_target,
        )
        self.catboost_ = CatBoostClassifier(
            iterations=600, learning_rate=0.03, depth=5,
            cat_features=CAT_COLS, eval_metric="Accuracy", random_seed=42,
            verbose=0,
        )
        self.catboost_.fit(
            base.iloc[fit], base_target[fit],
            eval_set=(base.iloc[evaluation], base_target[evaluation]),
            early_stopping_rounds=50,
        )
        self.neural_ = NeuralProbability().fit(
            base_neural.iloc[fit], base_target[fit],
            X_val=base_neural.iloc[evaluation], y_val=base_target[evaluation],
            cat_dims=cat_dims,
        )
        self.meta_ = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000, random_state=42
        ).fit(self._meta_features(meta), meta_target)
        return self

    def predict(self, X):
        probability = self.meta_.predict_proba(self._meta_features(X))[:, 1]
        return (probability >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records the read without executing it.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark raw labels and features. The source first reserves a fixed 20%
    # for five-fold logistic OOF validation. Keep its exact split and row order
    # as metadata, then let the plan's splitter drive those five scored folds.
    metadata = data["Survived"].skb.apply_func(original_split_metadata)
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).assign(
        _base_training=metadata["_base_training"],
        _source_order=metadata["_source_order"],
    ).skb.mark_as_X(cv=MetaValidationCV(), split_kwargs={})

    # 3. Recorded preprocessing and models. Medians and ticket frequencies were
    #    originally learned on the full table; separate estimators now learn
    #    these statistics from each meta fold's training rows (fixed base
    #    rows plus meta-training rows), excluding its scored meta rows.
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
    )[FEATURE_COLS + ["_base_training", "_source_order"]]

    # Preserve CatBoost's string categorical inputs. The two helper columns
    # stay available to the splitter/ensemble and never enter a fitted model.
    model_features = features[CAT_COLS].astype(str).skb.concat(
        [features[NUM_COLS + ["_base_training", "_source_order"]]], axis=1
    )
    pred = model_features.skb.apply(LogitStackingClassifier(), y=y)

    # 4. Score the original five meta-validation folds. Grid search reports
    # their unweighted mean; the source pooled the 143 OOF predictions, so
    # unequal fold sizes can produce a small aggregation-only score gap.
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
