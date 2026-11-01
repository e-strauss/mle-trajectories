"""train0: CatBoost and the neural net, averaged with equal weights.

The original fits both models on one stratified training split and early-stops
CatBoost on the same validation rows it scores, after filling Age and Fare with
the full-table medians. That preparation and that split stay upstream of
``mark_as_X``. The validation rows are passed to both estimators as
``eval_set``.

Each model is its own ``.skb.apply``. CatBoost is a plain
``CatBoostClassifier``: a subclass is a different type, and Stratum's CatBoost
kernel matches ``CatBoostClassifier``. ``cat_features`` stays a fit argument
because the constructor form fails ``clone``. ``prediction_type="Probability"``
is a predict argument, so ``predict`` returns one column per class.
``classes_`` is ``[0, 1]``, and column 1 is the positive-class probability the
subclass used to return.

``VotingClassifier`` only forwards ``sample_weight``, so it cannot carry
``cat_features``, ``eval_set``, or ``early_stopping_rounds``. The two
classifier nodes feed ``MeanProbabilityEnsemble``, whose ``predict`` is the
equal-weight average thresholded at 0.5. That average is the whole computation
inside the ensemble.
"""

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

CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class TabularNeuralNet(nn.Module):
    def __init__(
        self,
        cat_dims,
        emb_dims,
        num_features,
        hidden_dims=(64, 32),
        dropout=0.2,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, embedding_dim)
                for num_categories, embedding_dim in zip(cat_dims, emb_dims)
            ]
        )

        in_dim = sum(emb_dims) + num_features
        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        combined = torch.cat(embeddings + [x_num], dim=1)
        return self.network(combined)


class TabularNeuralNetClassifier(ClassifierMixin, BaseEstimator):
    """The original torch model, without the CatBoost fit or the average."""

    def __init__(
        self,
        cat_cols=tuple(CAT_COLS),
        num_cols=tuple(NUM_COLS),
        random_seed=42,
        hidden_dims=(64, 32),
        dropout=0.2,
        nn_learning_rate=1e-2,
        weight_decay=1e-4,
        scheduler_factor=0.5,
        scheduler_patience=10,
        batch_size=32,
        num_epochs=150,
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.random_seed = random_seed
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.nn_learning_rate = nn_learning_rate
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def _prepare_inputs(self, X, fit=False):
        categorical_arrays = []

        if fit:
            self.label_encoders_ = {}

        for column in self.cat_cols:
            values = X[column].astype(str)

            if fit:
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(values)
                self.label_encoders_[column] = encoder
            else:
                encoder = self.label_encoders_[column]
                mapping = {
                    value: index for index, value in enumerate(encoder.classes_)
                }
                encoded = np.asarray(
                    [mapping.get(value, 0) for value in values.to_numpy()],
                    dtype=np.int64,
                )

            categorical_arrays.append(encoded)

        X_cat = np.stack(categorical_arrays, axis=1).astype(np.int64)

        X_num_raw = X.loc[:, list(self.num_cols)].to_numpy(dtype=np.float32)
        if fit:
            self.scaler_ = StandardScaler()
            X_num = self.scaler_.fit_transform(X_num_raw)
        else:
            X_num = self.scaler_.transform(X_num_raw)

        return X_cat, X_num.astype(np.float32)

    def fit(self, X, y, eval_set=None):
        if eval_set is None:
            raise ValueError("An eval_set is required for the original training logic.")

        X_val, y_val = eval_set
        y_train = np.asarray(y, dtype=np.int64)
        y_validation = np.asarray(y_val, dtype=np.int64)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        self.classes_ = np.unique(y_train)

        X_cat_train, X_num_train = self._prepare_inputs(X, fit=True)
        X_cat_val, X_num_val = self._prepare_inputs(X_val, fit=False)

        cat_dims = [
            len(self.label_encoders_[column].classes_) for column in self.cat_cols
        ]
        emb_dims = [
            max(2, int(round(category_count**0.5 * 2)))
            for category_count in cat_dims
        ]

        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.nn_model_ = TabularNeuralNet(
            cat_dims=cat_dims,
            emb_dims=emb_dims,
            num_features=X_num_train.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.nn_model_.parameters(),
            lr=self.nn_learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        t_X_cat_train = torch.tensor(
            X_cat_train, dtype=torch.long, device=self.device_
        )
        t_X_num_train = torch.tensor(
            X_num_train, dtype=torch.float32, device=self.device_
        )
        t_y_train = torch.tensor(y_train, dtype=torch.long, device=self.device_)

        t_X_cat_val = torch.tensor(X_cat_val, dtype=torch.long, device=self.device_)
        t_X_num_val = torch.tensor(
            X_num_val, dtype=torch.float32, device=self.device_
        )

        for _ in range(self.num_epochs):
            self.nn_model_.train()
            permutation = torch.randperm(t_X_cat_train.size(0), device=self.device_)

            for start in range(0, t_X_cat_train.size(0), self.batch_size):
                batch_indices = permutation[start : start + self.batch_size]
                batch_cat = t_X_cat_train[batch_indices]
                batch_num = t_X_num_train[batch_indices]
                batch_y = t_y_train[batch_indices]

                optimizer.zero_grad()
                outputs = self.nn_model_(batch_cat, batch_num)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            self.nn_model_.eval()
            with torch.no_grad():
                validation_outputs = self.nn_model_(t_X_cat_val, t_X_num_val)
                validation_predictions = (
                    torch.argmax(validation_outputs, dim=1).cpu().numpy()
                )
                validation_accuracy = accuracy_score(
                    y_validation, validation_predictions
                )

            scheduler.step(validation_accuracy)

        return self

    def predict_proba(self, X):
        X_cat, X_num = self._prepare_inputs(X, fit=False)
        t_X_cat = torch.tensor(X_cat, dtype=torch.long, device=self.device_)
        t_X_num = torch.tensor(X_num, dtype=torch.float32, device=self.device_)

        self.nn_model_.eval()
        with torch.no_grad():
            outputs = self.nn_model_(t_X_cat, t_X_num)
            return torch.softmax(outputs, dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def _positive_class_probability(values):
    """The positive-class column of one member.

    A plain ``CatBoostClassifier.predict(prediction_type="Probability")``
    returns shape ``(n_samples, n_classes)`` in ``classes_`` order. For this
    target that order is ``[0, 1]``, so column 1 is ``predict_proba[:, 1]``.
    The neural net's ``predict`` already returns that column.
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
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Full-table medians, matching the original preparation before its split.
    title = (
        train_df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Missing")
    )
    cabin_deck = train_df["Cabin"].str[0].fillna("Missing")
    prepared = train_df.assign(
        Title=title,
        FamilySize=train_df["SibSp"] + train_df["Parch"] + 1,
        CabinDeck=cabin_deck,
        Embarked=train_df["Embarked"].fillna("Missing"),
        Age=train_df["Age"].fillna(train_df["Age"].median()),
        Fare=train_df["Fare"].fillna(train_df["Fare"].median()),
    )
    model_features = prepared[FEATURE_COLS].assign(
        Pclass=prepared["Pclass"].astype(str),
        Sex=prepared["Sex"].astype(str),
        Embarked=prepared["Embarked"].astype(str),
        Title=prepared["Title"].astype(str),
        CabinDeck=prepared["CabinDeck"].astype(str),
    )

    # The original early-stops on the same validation rows it scores. Keep that
    # split here. mark_as_X / mark_as_y copy a node and keep its id, so the
    # split holds model_features and its own Survived lookup, and the marks
    # below are separate operations.
    split_target = train_df["Survived"]
    fixed_split = model_features.skb.apply_func(
        train_test_split,
        split_target,
        test_size=0.2,
        random_state=42,
        stratify=split_target,
    )
    X_validation = fixed_split[1]
    y_validation = fixed_split[3]

    y = train_df["Survived"].skb.mark_as_y()
    X = model_features[FEATURE_COLS].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=42,
        ),
        split_kwargs={},
    )

    # cat_features is a fit argument: the constructor form fails clone.
    # prediction_type stays a predict argument so the estimator is a plain
    # CatBoostClassifier, which is the class Stratum lowers. Column 1 of that
    # matrix is the positive-class probability. The ensemble node consumes
    # both classifier nodes and only averages them.
    cb_probability = X.skb.apply(
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
            "eval_set": (X_validation, y_validation),
            "early_stopping_rounds": 50,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    nn_probability = X.skb.apply(
        TabularNeuralNetClassifier(),
        y=y,
        fit_kwargs={"eval_set": (X_validation, y_validation)},
    )
    pred = cb_probability.skb.apply(
        MeanProbabilityEnsemble(),
        y=y,
        fit_kwargs={"nn_probability": nn_probability},
        predict_kwargs={"nn_probability": nn_probability},
    )


if __name__ == "__main__":
    search = pred.skb.make_grid_search(
        n_jobs=1,
        fitted=True,
        refit=False,
        scoring="accuracy",
    )
    print(search.results_)
    for variant_score in search.results_["mean_test_score"]:
        print(f"Variant score: {variant_score}")
    print(
        "Final Validation Performance: "
        f"{search.results_['mean_test_score'].iloc[0]}"
    )
