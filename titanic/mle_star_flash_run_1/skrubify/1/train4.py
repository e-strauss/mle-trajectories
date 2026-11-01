"""train4: CatBoost and a neural net, blended with a tuned weight and threshold.

The original builds features on the whole table, then fits both models on one
stratified hold-out and early-stops, schedules, and picks the blend on the same
validation rows it scores. That preparation and that split stay upstream of
``mark_as_X``.

Each model is its own ``.skb.apply``. CatBoost is a plain
``CatBoostClassifier``: a subclass is a different type, and Stratum's CatBoost
kernel matches ``CatBoostClassifier``. ``cat_features`` stays a fit argument
because the constructor form fails ``clone``. ``prediction_type="Probability"``
makes ``predict`` return one column per class; column 1 is the positive-class
probability. The net is ``TabularNeuralNetClassifier``.

``VotingClassifier`` only forwards ``sample_weight``, so it cannot carry
``cat_features``, ``eval_set``, or ``early_stopping_rounds``. The blend is also
not a fixed soft vote: ``TunedProbabilityEnsemble.fit`` searches the weight and
the threshold on the hold-out probabilities of the two fitted members.
``predict`` applies that pair. The members are the two
``.skb.applied_estimator`` lookups: each one evaluates its apply node and
returns the fitted estimator.
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

CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_categories, embedding_dim)
             for num_categories, embedding_dim in zip(cat_dims, emb_dims)]
        )
        layers = []
        previous = sum(emb_dims) + num_features
        for width in hidden_dims:
            layers.extend([
                nn.Linear(previous, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            previous = width
        layers.append(nn.Linear(previous, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class TabularNeuralNetClassifier(ClassifierMixin, BaseEstimator):
    """The original torch loop, without the CatBoost fit or the blend."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def _prepare_inputs(self, X, fit=False):
        categorical_arrays = []
        if fit:
            self.label_encoders_ = {}
        for column in CAT_COLS:
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
                encoded = np.array(
                    [mapping.get(value, 0) for value in values.to_numpy()]
                )
            categorical_arrays.append(encoded)

        categories = np.stack(categorical_arrays, axis=1).astype(np.int64)
        numeric = X.loc[:, NUM_COLS].to_numpy(dtype=np.float32)
        if fit:
            self.scaler_ = StandardScaler()
            numeric = self.scaler_.fit_transform(numeric)
        else:
            numeric = self.scaler_.transform(numeric)
        return categories, numeric.astype(np.float32)

    def fit(self, X, y, eval_set=None):
        if eval_set is None:
            raise ValueError("An eval_set is required for the original training logic.")

        X_val, y_val = eval_set
        y_train = np.asarray(y).reshape(-1)
        y_validation = np.asarray(y_val).reshape(-1)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y_train)

        X_cat_train, X_num_train = self._prepare_inputs(X, fit=True)
        X_cat_val, X_num_val = self._prepare_inputs(X_val, fit=False)
        cat_dims = [len(self.label_encoders_[column].classes_) for column in CAT_COLS]
        emb_dims = [max(2, int(round(dim ** 0.5 * 2))) for dim in cat_dims]

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn_model_ = TabularNeuralNet(
            cat_dims, emb_dims, num_features=X_num_train.shape[1]
        ).to(self.device_)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.nn_model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        t_X_cat_train = torch.tensor(
            X_cat_train, dtype=torch.long, device=self.device_
        )
        t_X_num_train = torch.tensor(
            X_num_train, dtype=torch.float32, device=self.device_
        )
        t_y_train = torch.tensor(y_train, dtype=torch.long, device=self.device_)
        t_X_cat_val = torch.tensor(
            X_cat_val, dtype=torch.long, device=self.device_
        )
        t_X_num_val = torch.tensor(
            X_num_val, dtype=torch.float32, device=self.device_
        )

        for _ in range(150):
            self.nn_model_.train()
            permutation = torch.randperm(t_X_cat_train.size(0))
            for start in range(0, t_X_cat_train.size(0), 32):
                batch = permutation[start:start + 32]
                optimizer.zero_grad()
                outputs = self.nn_model_(t_X_cat_train[batch], t_X_num_train[batch])
                loss = criterion(outputs, t_y_train[batch])
                loss.backward()
                optimizer.step()

            self.nn_model_.eval()
            with torch.no_grad():
                val_outputs = self.nn_model_(t_X_cat_val, t_X_num_val)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_validation, val_preds))
        return self

    def predict_proba(self, X):
        categories, numeric = self._prepare_inputs(X, fit=False)
        cat_tensor = torch.tensor(
            categories, dtype=torch.long, device=self.device_
        )
        num_tensor = torch.tensor(
            numeric, dtype=torch.float32, device=self.device_
        )
        self.nn_model_.eval()
        with torch.no_grad():
            outputs = self.nn_model_(cat_tensor, num_tensor)
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


class TunedProbabilityEnsemble(ClassifierMixin, BaseEstimator):
    """Search a weight and threshold on the hold-out, then apply them.

    The members are the upstream classifier nodes. This estimator does not fit
    them. ``X`` is the CatBoost probability stream and ``nn_probability`` is
    the neural net's; ``predict`` blends those two vectors with the weight and
    threshold chosen in ``fit``. The search calls ``predict_proba`` on the
    fitted members at ``X_val``, which is the same hold-out the original blend
    used.
    """

    def fit(self, X, y, cb_model=None, nn_model=None, X_val=None, y_val=None):
        if cb_model is None or nn_model is None or X_val is None or y_val is None:
            raise ValueError(
                "cb_model, nn_model, X_val, and y_val are required to choose "
                "the blend."
            )
        del X  # training-fold probabilities; the search scores the hold-out
        y_validation = np.asarray(y_val).reshape(-1)
        cb_val_probs = _positive_class_probability(cb_model.predict_proba(X_val))
        nn_val_probs = _positive_class_probability(nn_model.predict_proba(X_val))
        best_score = -1.0
        self.weight_ = 0.5
        self.threshold_ = 0.5
        for weight in np.linspace(0.0, 1.0, 101):
            blended = weight * cb_val_probs + (1.0 - weight) * nn_val_probs
            for threshold in np.linspace(0.1, 0.9, 81):
                labels = (blended >= threshold).astype(int)
                score = accuracy_score(y_validation, labels)
                if score > best_score:
                    best_score = score
                    self.weight_ = weight
                    self.threshold_ = threshold
        self.classes_ = np.unique(np.asarray(y).reshape(-1))
        return self

    def predict(self, X, nn_probability=None):
        cb_probability = _positive_class_probability(X)
        nn_probability = _positive_class_probability(nn_probability)
        blended = (
            self.weight_ * cb_probability + (1.0 - self.weight_) * nn_probability
        )
        return (blended >= self.threshold_).astype(int)


with skrub.config_context(eager_data_ops=False):
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Full-table statistics, matching prepare_features before its split.
    raw_title = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = (
        raw_title.map({
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
        })
        .fillna("Rare")
        .where(raw_title.notna(), "Missing")
    )
    age_median = data.groupby(["Pclass", "Sex"])["Age"].transform("median")
    age = data["Age"].fillna(age_median).fillna(data["Age"].median())
    family_size = data["SibSp"] + data["Parch"] + 1
    family_tier = (
        family_size
        .where(family_size == 1, "SmallFamily")
        .where(family_size <= 4, "LargeFamily")
        .where(family_size != 1, "IsAlone")
    )
    ticket_frequency = data["Ticket"].map(data["Ticket"].value_counts()).fillna(1)
    prepared = data.assign(
        Title=title,
        Age=age,
        Sex_Pclass=data["Sex"].astype(str) + "_" + data["Pclass"].astype(str),
        FamilySize=family_size,
        FamilyTier=family_tier,
        HasCabin=data["Cabin"].notna().astype(int),
        TicketFreq=ticket_frequency,
        Embarked=data["Embarked"].fillna("Missing"),
        Fare=data["Fare"].fillna(data["Fare"].median()),
    )
    model_features = prepared[FEATURE_COLS]
    for column in CAT_COLS:
        model_features = model_features.assign(
            **{column: model_features[column].astype(str)}
        )

    # The original scores the same hold-out it uses for early stopping, the
    # scheduler, and the blend. Keep that split here. The marks below are
    # separate nodes, so this split stays on the full table.
    split_target = data["Survived"]
    fixed_split = model_features.skb.apply_func(
        train_test_split,
        split_target,
        test_size=0.2,
        random_state=42,
        stratify=split_target,
    )
    X_validation = fixed_split[1]
    y_validation = fixed_split[3]

    y = data["Survived"].skb.mark_as_y()
    X = model_features[FEATURE_COLS].skb.mark_as_X(
        cv=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )
    # cat_features is a fit argument: the constructor form fails clone.
    # prediction_type stays a predict argument so the estimator is a plain
    # CatBoostClassifier, which is the class Stratum lowers. Column 1 of that
    # matrix is the positive-class probability.
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
        TabularNeuralNetClassifier(random_state=42),
        y=y,
        fit_kwargs={"eval_set": (X_validation, y_validation)},
    )
    # applied_estimator evaluates each apply node and returns the fitted
    # member. The ensemble searches the hold-out with those models, then
    # thresholds the probability streams.
    pred = cb_probability.skb.apply(
        TunedProbabilityEnsemble(),
        y=y,
        fit_kwargs={
            "cb_model": cb_probability.skb.applied_estimator,
            "nn_model": nn_probability.skb.applied_estimator,
            "X_val": X_validation,
            "y_val": y_validation,
        },
        predict_kwargs={"nn_probability": nn_probability},
    )


if __name__ == "__main__":
    search = pred.skb.make_grid_search(
        n_jobs=1, fitted=True, refit=False, scoring="accuracy"
    )
    print(search.results_)
    for variant_score in search.results_["mean_test_score"]:
        print(f"Variant score: {variant_score}")
    print(
        "Final Validation Performance: "
        f"{search.results_['mean_test_score'].iloc[0]}"
    )
