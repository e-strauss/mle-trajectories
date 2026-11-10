import math

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


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConditionalAgeImputer(TransformerMixin, BaseEstimator):
    """Learn group medians and the post-group-imputation fallback median."""

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        group_values = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        self.fallback_ = X["Age"].fillna(group_values).median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Sex"]])
        group_values = pd.Series(
            self.group_medians_.reindex(keys).to_numpy(), index=X.index
        )
        return X.assign(Age=X["Age"].fillna(group_values).fillna(self.fallback_))


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
    """Create an early-stopping split from the outer fold's training rows."""

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


class NumericalTokenizer(nn.Module):
    def __init__(self, num_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(num_features, d_token) * (1.0 / math.sqrt(d_token))
        )
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))

    def forward(self, x_num):
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    def __init__(
        self,
        cat_dims,
        num_features,
        d_token=32,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        use_num_skip=True,
    ):
        super().__init__()
        self.use_num_skip = use_num_skip
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(num_c, d_token) for num_c in cat_dims]
        )
        self.num_tokenizer = NumericalTokenizer(num_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 2),
        )
        if self.use_num_skip:
            self.num_skip = nn.Linear(num_features, 2)

    def forward(self, x_cat, x_num):
        batch_size = x_cat.size(0)
        tokens = [self.cls_token.expand(batch_size, -1, -1)]
        cat_tokens = [
            emb(x_cat[:, i]).unsqueeze(1)
            for i, emb in enumerate(self.cat_embeddings)
        ]
        tokens.append(torch.cat(cat_tokens, dim=1))
        tokens.append(self.num_tokenizer(x_num))
        x = self.transformer(torch.cat(tokens, dim=1))
        logits = self.head(x[:, 0, :])
        if self.use_num_skip:
            logits = logits + self.num_skip(x_num)
        return logits


class ClassicTabularMLP(nn.Module):
    def __init__(self, cat_dims, num_features, hidden_dims=(64, 32), dropout=0.2):
        super().__init__()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]
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
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.network(torch.cat(embs + [x_num], dim=1))


class NeuralPositiveProbability(ClassifierMixin, BaseEstimator):
    """Only model construction, tensor conversion, and the original torch loop."""

    def __init__(
        self,
        backbone="ft",
        use_num_skip=True,
        epochs=120,
        lr=3e-3,
        random_state=42,
    ):
        self.backbone = backbone
        self.use_num_skip = use_num_skip
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        # The source explicitly calls set_seed(42) before each architecture.
        set_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_features_in_ = X.shape[1]

        cat_values = X[CAT_COLS].to_numpy(dtype=np.int64)
        num_values = X[NUM_COLS].to_numpy(dtype=np.float32)
        # Encoders are fitted separately in the plan. Their contiguous training
        # codes determine embedding sizes, just as len(le.classes_) did.
        cat_dims = (cat_values.max(axis=0) + 1).tolist()
        if self.backbone == "ft":
            self.model_ = FTTransformer(
                cat_dims,
                num_features=len(NUM_COLS),
                use_num_skip=self.use_num_skip,
            )
        else:
            self.model_ = ClassicTabularMLP(
                cat_dims, num_features=len(NUM_COLS)
            )
        self.model_ = self.model_.to(self.device_)

        x_cat = torch.tensor(cat_values, dtype=torch.long, device=self.device_)
        x_num = torch.tensor(num_values, dtype=torch.float32, device=self.device_)
        target = torch.tensor(
            np.asarray(y), dtype=torch.long, device=self.device_
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=1e-4
        )

        batch_size = 32
        for epoch in range(self.epochs):
            self.model_.train()
            permutation = torch.randperm(x_cat.size(0))
            for i in range(0, x_cat.size(0), batch_size):
                indices_batch = permutation[i : i + batch_size]
                optimizer.zero_grad()
                outputs = self.model_(
                    x_cat[indices_batch], x_num[indices_batch]
                )
                loss = criterion(outputs, target[indices_batch])
                loss.backward()
                optimizer.step()
        self.model_.eval()
        return self

    def predict_proba(self, X):
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
            return torch.softmax(self.model_(x_cat, x_num), dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def blend_and_threshold(cb_probs, neural_probs, mode):
    if mode == "fit":
        return None
    # prediction_type="Probability" returns both classes. Column 1 is the
    # positive class because Survived labels are 0 and 1.
    cb_probs = np.asarray(cb_probs, dtype=float)[:, 1]
    return (0.5 * cb_probs + 0.5 * neural_probs >= 0.5).astype(int)


def threshold(neural_probs, mode):
    if mode == "fit":
        return None
    return (neural_probs >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read; importing this file does not read or fit.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and features. This reproduces the original
    #    stratified train_test_split's exact outer training/validation rows.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, with each learned statistic in its own
    #    estimator. The original learned age medians, ticket frequencies, and
    #    the fare median before splitting; these now use training rows only.
    raw_title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = raw_title.map(
        {
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
        }
    ).fillna("Rare").where(raw_title.notna(), "Missing")

    age = X[["Pclass", "Sex", "Age"]].skb.apply(ConditionalAgeImputer())
    tickets = X[["Ticket"]].skb.apply(TicketFrequency())
    fare = X[["Fare"]].skb.apply(SimpleImputer(strategy="median"))

    features = X.assign(Title=title)
    features = features.assign(Age=age["Age"])
    features = features.assign(
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str)
    )
    features = features.assign(FamilySize=X["SibSp"] + X["Parch"] + 1)
    family_size = features["FamilySize"]
    # A row-shaped condition is required by pandas.where. The final override
    # preserves the original size == 1 first branch.
    family_tier = (
        family_size.le(4)
        .map({True: "SmallFamily", False: "LargeFamily"})
        .where(family_size != 1, "IsAlone")
    )
    features = features.assign(FamilyTier=family_tier)
    features = features.assign(HasCabin=X["Cabin"].notna().astype(int))
    features = features.assign(TicketFreq=tickets["TicketFreq"])
    features = features.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.assign(Fare=fare["Fare"])

    cats = features[CAT_COLS].astype(str)
    nums = features[NUM_COLS]
    cb_features = cats.skb.concat([nums], axis=1)

    # Sorted ordinal codes match the separate LabelEncoders. OrdinalEncoder
    # requires an unknown sentinel distinct from every known code: use -1,
    # then explicitly map it to the original unseen-category fallback of zero.
    nn_cats = cats.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
    )
    nn_cats = nn_cats.replace(-1, 0)
    nn_nums = nums.astype(np.float32).skb.apply(StandardScaler())
    nn_features = nn_cats.skb.concat([nn_nums], axis=1)

    # 4. Models. CatBoost originally early-stopped on the scored hold-out:
    #    keep its patience and iterations, but carve its eval set out of the
    #    outer training rows instead, avoiding that validation leakage.
    cb_split = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
    cb_probs = cb_split["X"].skb.apply(
        CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_split.get("y", y),
        fit_kwargs={
            "cat_features": CAT_COLS,
            "eval_set": (cb_split["X_val"], cb_split["y_val"]),
            "early_stopping_rounds": 50,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    ft_probs_base = nn_features.skb.apply(
        NeuralPositiveProbability(use_num_skip=True), y=y
    )
    ft_probs_no_skip = nn_features.skb.apply(
        NeuralPositiveProbability(use_num_skip=False), y=y
    )
    mlp_probs = nn_features.skb.apply(
        NeuralPositiveProbability(backbone="mlp", lr=1e-2), y=y
    )

    # 5. Exactly the four original variants, with the original probability
    #    averaging and >= 0.5 decoding. Gate arithmetic in estimator-fit mode.
    baseline = cb_probs.skb.apply_func(
        blend_and_threshold, ft_probs_base, skrub.eval_mode()
    )
    ablation_1 = cb_probs.skb.apply_func(
        blend_and_threshold, ft_probs_no_skip, skrub.eval_mode()
    )
    ablation_2 = cb_probs.skb.apply_func(
        blend_and_threshold, mlp_probs, skrub.eval_mode()
    )
    ablation_3 = ft_probs_base.skb.apply_func(threshold, skrub.eval_mode())

    pred = skrub.choose_from(
        {
            "Baseline (CatBoost + FT-Transformer w/ Skip)": baseline,
            "Ablation 1 (FT-Transformer without Numerical Skip)": ablation_1,
            "Ablation 2 (Classic Tabular MLP instead of FT-Transformer)": ablation_2,
            "Ablation 3 (Standalone FT-Transformer - No CatBoost)": ablation_3,
        },
        name="variant",
    ).as_data_op()

    # 6. Score all variants with accuracy; the splitter on mark_as_X drives.
    #    Drop the original progress and derived impact-summary reporting.
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
