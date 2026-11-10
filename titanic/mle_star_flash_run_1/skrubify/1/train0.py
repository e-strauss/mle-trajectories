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


CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
FEATURE_COLS = CAT_COLS + NUM_COLS


class GetXY(TransformerMixin, BaseEstimator):
    """Reserve an internal validation set from this fold's training rows."""

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
        prev_dim = sum(emb_dims) + num_features
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(embeddings + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Keep only the original torch model and training loop in this estimator."""

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

    def fit(self, X, y, eval_set=None, cat_dims=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Cardinalities are fit-only statistics supplied by recorded nodes.
        cat_dims = [int(dim) for dim in np.asarray(cat_dims)]
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

        X_val, y_val = eval_set
        train_cat = torch.tensor(
            X[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        train_num = torch.tensor(
            X[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        train_y = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        val_cat = torch.tensor(
            X_val[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        val_num = torch.tensor(
            X_val[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )

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
                outputs = self.model_(val_cat, val_num)
                val_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                val_accuracy = accuracy_score(y_val, val_predictions)
            scheduler.step(val_accuracy)

        # As in the original, use the last epoch, not a best-epoch checkpoint.
        self.model_.eval()
        return self

    def predict(self, X):
        cat = torch.tensor(
            X[CAT_COLS].to_numpy(), dtype=torch.long, device=self.device_
        )
        num = torch.tensor(
            X[NUM_COLS].to_numpy(), dtype=torch.float32, device=self.device_
        )
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(cat, num)
            return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()


def blend_predictions(cb_probability, nn_probability, mode):
    # Predictor nodes contain fitted estimators, not probabilities, in fit mode.
    # prediction_type="Probability" returns (n_samples, n_classes) in classes_
    # order. classes_ is [0, 1], so column 1 is the positive-class probability
    # the net's predict already returns.
    if mode == "fit":
        return None
    cb_probability = np.asarray(cb_probability, dtype=float)[:, 1]
    return (0.5 * cb_probability + 0.5 * nn_probability >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read without reading anything on import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — mark the raw target. StratifiedShuffleSplit reproduces
    #    the original stratified train_test_split's exact outer hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, retaining the original feature order.
    features = X.assign(
        Title=X["Name"].str.extract(
            r" ([A-Za-z]+)\.", expand=False
        ).fillna("Missing")
    )
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    features = features.assign(
        CabinDeck=features["Cabin"].str[0].fillna("Missing")
    )
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing")
    )
    features = features[FEATURE_COLS]

    # The original full-table median imputation leaked into validation.
    # Learn these two medians on each outer fold's training rows instead.
    imputed = features[["Age", "Fare"]].skb.apply(
        SimpleImputer(strategy="median").set_output(transform="pandas")
    )
    features = features.assign(Age=imputed["Age"], Fare=imputed["Fare"])

    categorical = features[CAT_COLS].astype(str)
    cb_features = categorical.skb.concat([features[NUM_COLS]], axis=1)
    cb_features = cb_features[FEATURE_COLS]

    # 4. CatBoost — preserve early stopping, exposing its internal split.
    #    The original early-stopped on the scored hold-out. To avoid that leak,
    #    reserve 20% of the outer training rows, retaining stratification, seed,
    #    patience, and the original 600-iteration limit.
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

    # 5. Neural preprocessing — sorted string categories match LabelEncoder.
    #    OrdinalEncoder requires an unknown sentinel distinct from known codes:
    #    encode unknowns as -1, then replace them with the original fallback 0.
    nn_categorical = categorical.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        ).set_output(transform="pandas")
    )
    nn_categorical = nn_categorical.replace(-1, 0)

    # Encoding and scaling remain separate, fold-fitted plan operators.
    nn_numeric = features[NUM_COLS].astype(np.float32).skb.apply(
        StandardScaler().set_output(transform="pandas")
    )
    nn_features = nn_categorical.skb.concat([nn_numeric], axis=1)
    nn_features = nn_features[FEATURE_COLS]

    # This statistic is consumed only during fitting. Computing it here keeps
    # it visible and restricted to the outer fold's training vocabulary.
    cat_dims = (nn_categorical.max(axis=0) + 1).to_numpy()

    # 6. Neural training — the original scheduler also consulted the scored
    #    hold-out every epoch. Use the same internal stratified 20% reservation
    #    as CatBoost instead; the outer scored rows remain untouched.
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
            "cat_dims": cat_dims,
        },
    )

    # 7. Preserve the original equal probability blend and inclusive threshold.
    pred = cb_probability.skb.apply_func(
        blend_predictions, nn_probability, skrub.eval_mode()
    )

    # 8. Score the single original ensemble, not its individual components.
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
