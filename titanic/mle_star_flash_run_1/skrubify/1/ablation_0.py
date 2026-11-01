"""ablation_0: ensemble, CatBoost, the neural net, and the same models without the extra features.

The original fills Age and Fare with the full-table medians, then early-stops
CatBoost and steps the neural-net schedule on the same 143 validation rows it
scores. That preparation and that split stay upstream of ``mark_as_X``. The
validation rows are passed to both estimators as ``eval_set``. The net trains
on every training-fold row.

The engineered columns and the raw columns are separate applies: one CatBoost
and one neural net each. ``pred`` chooses among the four scored outputs of
those applies. CatBoost is a plain ``CatBoostClassifier``. ``cat_features``
stays a fit argument because the constructor form fails ``clone``.
``prediction_type="Probability"`` is a predict argument, so ``predict``
returns one column per class. ``classes_`` is ``[0, 1]``, and column 1 is
the positive-class probability.

The original fits the feature-engineered models once and reuses those
probabilities for the ensemble, CatBoost-only, and neural-net-only scores,
then fits the raw models once. ``__main__`` does the same two fits. The four
variants remain in ``pred``.
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

FE_CAT = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"]
FE_NUM = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
RAW_CAT = ["Pclass", "Sex", "Embarked"]
RAW_NUM = ["Age", "SibSp", "Parch", "Fare"]


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=(64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_c, e_dim)
                for num_c, e_dim in zip(cat_dims, emb_dims)
            ]
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
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embs + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """The original torch loop. Encoding and scaling happen inside the fit."""

    def __init__(self, cat_cols, num_cols, random_state=42):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.random_state = random_state

    def _encode(self, X, fit):
        categorical = []
        if fit:
            self.encoders_ = {}
        for column in self.cat_cols:
            values = X[column].astype(str)
            if fit:
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(values)
                self.encoders_[column] = encoder
            else:
                encoder = self.encoders_[column]
                mapping = {
                    value: index
                    for index, value in enumerate(encoder.classes_)
                }
                encoded = np.array(
                    [mapping.get(value, 0) for value in values.to_numpy()]
                )
            categorical.append(encoded)
        categories = np.stack(categorical, axis=1).astype(np.int64)
        raw_numeric = X.loc[:, list(self.num_cols)].to_numpy().astype(np.float32)
        if fit:
            self.scaler_ = StandardScaler()
            numeric = self.scaler_.fit_transform(raw_numeric)
        else:
            numeric = self.scaler_.transform(raw_numeric)
        return categories, numeric

    def fit(self, X, y, eval_set=None):
        if eval_set is None:
            raise ValueError("An eval_set is required for the original training logic.")
        X_val, y_val = eval_set
        y_train = np.asarray(y)
        y_validation = np.asarray(y_val)

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y_train)

        train_cat, train_num = self._encode(X, fit=True)
        val_cat, val_num = self._encode(X_val, fit=False)
        cat_dims = [
            len(self.encoders_[column].classes_) for column in self.cat_cols
        ]
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_ = TabularNeuralNet(
            cat_dims, emb_dims, num_features=train_num.shape[1]
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=1e-2, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        train_cat = torch.tensor(
            train_cat, dtype=torch.long, device=self.device_
        )
        train_num = torch.tensor(
            train_num, dtype=torch.float32, device=self.device_
        )
        train_y = torch.tensor(y_train, dtype=torch.long, device=self.device_)
        val_cat = torch.tensor(val_cat, dtype=torch.long, device=self.device_)
        val_num = torch.tensor(val_num, dtype=torch.float32, device=self.device_)

        for _ in range(150):
            self.model_.train()
            permutation = torch.randperm(train_cat.size(0))
            for start in range(0, train_cat.size(0), 32):
                indices_batch = permutation[start : start + 32]
                optimizer.zero_grad()
                outputs = self.model_(
                    train_cat[indices_batch], train_num[indices_batch]
                )
                loss = criterion(outputs, train_y[indices_batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(val_cat, val_num)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            scheduler.step(accuracy_score(y_validation, val_preds))

        self.model_.eval()
        return self

    def predict(self, X):
        categories, numeric = self._encode(X, fit=False)
        x_cat = torch.tensor(
            categories, dtype=torch.long, device=self.device_
        )
        x_num = torch.tensor(numeric, dtype=torch.float32, device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(x_cat, x_num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def _positive_class_probability(values):
    """Column 1 when ``predict`` returned the class-probability matrix."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        return array[:, 1]
    return np.reshape(array, -1)


def threshold_probability(probability, mode):
    if mode == "fit":
        return None
    return (_positive_class_probability(probability) >= 0.5).astype(int)


# One ensemble fit produces both probability vectors. The runner scores the
# CatBoost-only and neural-net-only variants from the scored-fold entry.
_HELD = []


def threshold_ensemble(cb_probability, nn_probability, mode):
    if mode == "fit":
        return None
    cb = _positive_class_probability(cb_probability)
    nn = _positive_class_probability(nn_probability)
    labels = ((0.5 * cb + 0.5 * nn) >= 0.5).astype(int)
    _HELD.append((cb, nn, labels))
    return labels


def _model_frame(frame, columns, cat_cols):
    selected = frame[columns]
    for column in cat_cols:
        selected = selected.assign(**{column: selected[column].astype(str)})
    return selected


def _holdout(frame, target):
    parts = frame.skb.apply_func(
        train_test_split,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    return parts[1], parts[3]


def _probabilities(features, cat_cols, num_cols, y, X_val, y_val):
    cb_probability = features.skb.apply(
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
            "cat_features": list(cat_cols),
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 50,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    nn_probability = features.skb.apply(
        NeuralPositiveProbability(cat_cols=list(cat_cols), num_cols=list(num_cols)),
        y=y,
        fit_kwargs={"eval_set": (X_val, y_val)},
    )
    return cb_probability, nn_probability


with skrub.config_context(eager_data_ops=False):
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Full-table medians, matching the original preparation before its split.
    title = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Missing")
    prepared = data.assign(
        Title=title,
        FamilySize=data["SibSp"] + data["Parch"] + 1,
        CabinDeck=data["Cabin"].str[0].fillna("Missing"),
        Embarked=data["Embarked"].fillna("Missing"),
        Age=data["Age"].fillna(data["Age"].median()),
        Fare=data["Fare"].fillna(data["Fare"].median()),
    )
    fe_frame = _model_frame(prepared, FE_CAT + FE_NUM, FE_CAT)
    raw_frame = _model_frame(prepared, RAW_CAT + RAW_NUM, RAW_CAT)

    # The original early-stops on the same validation rows it scores.
    split_target = data["Survived"]
    fe_X_val, fe_y_val = _holdout(fe_frame, split_target)
    raw_X_val, raw_y_val = _holdout(raw_frame, split_target)

    y = data["Survived"].skb.mark_as_y()
    X = fe_frame.skb.mark_as_X(
        cv=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )
    X_raw = X[RAW_CAT + RAW_NUM]

    cb_probs, nn_probs = _probabilities(
        X, FE_CAT, FE_NUM, y, fe_X_val, fe_y_val
    )
    cb_raw_probs, nn_raw_probs = _probabilities(
        X_raw, RAW_CAT, RAW_NUM, y, raw_X_val, raw_y_val
    )

    mode = skrub.eval_mode()
    fe_ensemble = cb_probs.skb.apply_func(threshold_ensemble, nn_probs, mode)
    raw_ensemble = cb_raw_probs.skb.apply_func(
        threshold_ensemble, nn_raw_probs, mode
    )
    pred = skrub.choose_from(
        {
            "Baseline (Full Ensemble + FE)": fe_ensemble,
            "Ablation 1 (CatBoost Only - No NN)": cb_probs.skb.apply_func(
                threshold_probability, mode
            ),
            "Ablation 2 (NN Only - No CatBoost)": nn_probs.skb.apply_func(
                threshold_probability, mode
            ),
            "Ablation 3 (No Feature Engineering)": raw_ensemble,
        },
        name="variant",
    ).as_data_op()


def _captured_probabilities(captured, y_true, grid_score):
    """The scored-fold probabilities saved while the ensemble was predicted."""
    y_true = np.asarray(y_true).reshape(-1)
    matches = []
    for cb, nn, labels in captured:
        labels = np.asarray(labels).reshape(-1)
        if labels.shape != y_true.shape:
            continue
        accuracy = float(np.mean(labels == y_true))
        if abs(accuracy - float(grid_score)) <= 1e-12:
            matches.append((cb, nn))
    if len(matches) != 1:
        lengths = [int(np.asarray(labels).reshape(-1).size) for _, _, labels in captured]
        raise RuntimeError(
            f"expected one capture of the scored rows, found {len(matches)} "
            f"of {len(captured)} (lengths {lengths})"
        )
    return matches[0]


if __name__ == "__main__":
    def _fit_ensemble(node):
        start = len(_HELD)
        search = node.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        score = float(search.results_["mean_test_score"].iloc[0])
        return score, _HELD[start:]

    baseline_score, fe_captured = _fit_ensemble(fe_ensemble)
    y_fe = fe_y_val.skb.eval()
    cb_fe, nn_fe = _captured_probabilities(fe_captured, y_fe, baseline_score)
    cb_only_score = accuracy_score(y_fe, (np.asarray(cb_fe) >= 0.5).astype(int))
    nn_only_score = accuracy_score(y_fe, (np.asarray(nn_fe) >= 0.5).astype(int))

    no_fe_score, raw_captured = _fit_ensemble(raw_ensemble)
    _captured_probabilities(raw_captured, raw_y_val.skb.eval(), no_fe_score)

    ablation_results = {
        "Baseline (Full Ensemble + FE)": baseline_score,
        "Ablation 1 (CatBoost Only - No NN)": cb_only_score,
        "Ablation 2 (NN Only - No CatBoost)": nn_only_score,
        "Ablation 3 (No Feature Engineering)": no_fe_score,
    }
    print("=== Ablation Study Results ===")
    for name, score in ablation_results.items():
        delta = score - baseline_score
        print(f"{name:40s} | Validation Accuracy: {score:.5f} | Delta: {delta:+.5f}")

    drops = {
        "CatBoost Model": baseline_score - nn_only_score,
        "Neural Network Model": baseline_score - cb_only_score,
        "Feature Engineering": baseline_score - no_fe_score,
    }
    most_important = max(drops, key=drops.get)
    print("\n=== Ablation Summary ===")
    print(
        f"The component contributing the most to overall performance is '{most_important}' "
        f"with a performance drop of {drops[most_important]:.5f} when removed/altered."
    )
