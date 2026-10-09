import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load training data
train_df = pd.read_csv("./input/train.csv")


def prepare_features_baseline(df):
    df = df.copy()

    # Standardized Title categories
    raw_title = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    df["Title"] = raw_title.map(
        lambda x: title_mapping.get(x, "Rare") if pd.notna(x) else "Missing"
    )

    # Impute missing Age using grouped median
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Sociological interaction feature
    df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)

    # Family dynamics
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    def assign_family_tier(size):
        if size == 1:
            return "IsAlone"
        elif size <= 4:
            return "SmallFamily"
        else:
            return "LargeFamily"

    df["FamilyTier"] = df["FamilySize"].apply(assign_family_tier)

    # Indicators & frequencies
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["TicketFreq"] = df["Ticket"].map(df["Ticket"].value_counts()).fillna(1)

    # Missing value handling
    df["Embarked"] = df["Embarked"].fillna("Missing")
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    cat_cols = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
    num_cols = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]

    return df, cat_cols, num_cols


def prepare_features_no_interactions(df):
    """Ablation 1: Remove composite sociological and engineered interaction features."""
    df = df.copy()

    raw_title = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    df["Title"] = raw_title.map(
        lambda x: title_mapping.get(x, "Rare") if pd.notna(x) else "Missing"
    )

    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Embarked"] = df["Embarked"].fillna("Missing")
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    cat_cols = ["Pclass", "Sex", "Embarked", "Title"]
    num_cols = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin"]

    return df, cat_cols, num_cols


def prepare_features_global_imputation(df):
    """Ablation 2: Replace conditional/grouped imputation with simple global median imputation."""
    df = df.copy()

    raw_title = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    df["Title"] = raw_title.map(
        lambda x: title_mapping.get(x, "Rare") if pd.notna(x) else "Missing"
    )

    # Simple global imputation
    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    def assign_family_tier(size):
        if size == 1:
            return "IsAlone"
        elif size <= 4:
            return "SmallFamily"
        else:
            return "LargeFamily"

    df["FamilyTier"] = df["FamilySize"].apply(assign_family_tier)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["TicketFreq"] = df["Ticket"].map(df["Ticket"].value_counts()).fillna(1)
    df["Embarked"] = df["Embarked"].fillna("Missing")
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    cat_cols = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
    num_cols = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]

    return df, cat_cols, num_cols


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=[64, 32]):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_c, e_dim) for num_c, e_dim in zip(cat_dims, emb_dims)]
        )
        total_emb_dim = sum(emb_dims)
        in_dim = total_emb_dim + num_features

        layers = []
        prev_dim = in_dim
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
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.network(x)


def train_and_eval_pipeline(df_processed, cat_cols, num_cols, y, idx_train, idx_val):
    # 1. CatBoost
    X_cb = df_processed[cat_cols + num_cols].copy()
    for col in cat_cols:
        X_cb[col] = X_cb[col].astype(str)

    X_cb_train, X_cb_val = X_cb.iloc[idx_train], X_cb.iloc[idx_val]
    y_cb_train, y_cb_val = y[idx_train], y[idx_val]

    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=5,
        cat_features=cat_cols,
        eval_metric="Accuracy",
        random_seed=42,
        verbose=0,
    )
    cb_model.fit(
        X_cb_train, y_cb_train, eval_set=(X_cb_val, y_cb_val), early_stopping_rounds=50
    )
    cb_val_probs = cb_model.predict_proba(X_cb_val)[:, 1]

    # 2. Neural Network
    cat_arrays_train, cat_arrays_val, cat_dims = [], [], []
    for col in cat_cols:
        le = LabelEncoder()
        cat_train_encoded = le.fit_transform(
            df_processed.iloc[idx_train][col].astype(str)
        )
        val_cats = df_processed.iloc[idx_val][col].astype(str).values
        mapping = {val: i for i, val in enumerate(le.classes_)}
        cat_val_encoded = np.array([mapping.get(v, 0) for v in val_cats])

        cat_arrays_train.append(cat_train_encoded)
        cat_arrays_val.append(cat_val_encoded)
        cat_dims.append(len(le.classes_))

    X_nn_cat_train = np.stack(cat_arrays_train, axis=1).astype(np.int64)
    X_nn_cat_val = np.stack(cat_arrays_val, axis=1).astype(np.int64)

    scaler = StandardScaler()
    X_nn_num_train = scaler.fit_transform(
        df_processed.iloc[idx_train][num_cols].values.astype(np.float32)
    )
    X_nn_num_val = scaler.transform(
        df_processed.iloc[idx_val][num_cols].values.astype(np.float32)
    )

    emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    nn_model = TabularNeuralNet(
        cat_dims, emb_dims, num_features=X_nn_num_train.shape[1]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    t_X_cat_train = torch.tensor(X_nn_cat_train, dtype=torch.long, device=device)
    t_X_num_train = torch.tensor(X_nn_num_train, dtype=torch.float32, device=device)
    t_y_train = torch.tensor(y[idx_train], dtype=torch.long, device=device)

    t_X_cat_val = torch.tensor(X_nn_cat_val, dtype=torch.long, device=device)
    t_X_num_val = torch.tensor(X_nn_num_val, dtype=torch.float32, device=device)

    batch_size = 32
    num_epochs = 150

    for epoch in range(num_epochs):
        nn_model.train()
        permutation = torch.randperm(t_X_cat_train.size(0))

        for i in range(0, t_X_cat_train.size(0), batch_size):
            indices_batch = permutation[i : i + batch_size]
            batch_cat = t_X_cat_train[indices_batch]
            batch_num = t_X_num_train[indices_batch]
            batch_y = t_y_train[indices_batch]

            optimizer.zero_grad()
            outputs = nn_model(batch_cat, batch_num)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        with torch.no_grad():
            val_outputs = nn_model(t_X_cat_val, t_X_num_val)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_acc = accuracy_score(y[idx_val], val_preds)

        scheduler.step(val_acc)

    nn_model.eval()
    with torch.no_grad():
        val_outputs = nn_model(t_X_cat_val, t_X_num_val)
        nn_val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()

    return cb_val_probs, nn_val_probs


# Split indices
y = train_df["Survived"].values
indices = np.arange(len(y))
idx_train, idx_val, _, y_val = train_test_split(
    indices, y, test_size=0.2, random_state=42, stratify=y
)

# Run Baseline
df_base, cat_base, num_base = prepare_features_baseline(train_df)
cb_probs_base, nn_probs_base = train_and_eval_pipeline(
    df_base, cat_base, num_base, y, idx_train, idx_val
)
baseline_50_50 = accuracy_score(
    y_val, ((0.5 * cb_probs_base + 0.5 * nn_probs_base) >= 0.5).astype(int)
)

# Run Ablation 1: Remove Interaction Features (Sex_Pclass, FamilyTier, TicketFreq)
df_no_int, cat_no_int, num_no_int = prepare_features_no_interactions(train_df)
cb_probs_no_int, nn_probs_no_int = train_and_eval_pipeline(
    df_no_int, cat_no_int, num_no_int, y, idx_train, idx_val
)
score_ablation_no_interactions = accuracy_score(
    y_val, ((0.5 * cb_probs_no_int + 0.5 * nn_probs_no_int) >= 0.5).astype(int)
)

# Run Ablation 2: Global Simple Imputation instead of Grouped Imputation
df_global, cat_global, num_global = prepare_features_global_imputation(train_df)
cb_probs_global, nn_probs_global = train_and_eval_pipeline(
    df_global, cat_global, num_global, y, idx_train, idx_val
)
score_ablation_global_imputation = accuracy_score(
    y_val, ((0.5 * cb_probs_global + 0.5 * nn_probs_global) >= 0.5).astype(int)
)

# Run Ablation 3: Weighted Ensembling (85% CatBoost + 15% NN) instead of 50/50
score_ablation_weighted_blend = accuracy_score(
    y_val, ((0.85 * cb_probs_base + 0.15 * nn_probs_base) >= 0.5).astype(int)
)

# Output Results
results = {
    "Baseline (50/50 Ensemble + Grouped Imputation + Interactions)": baseline_50_50,
    "Ablation 1 (Remove Interaction Features: Sex_Pclass, FamilyTier, TicketFreq)": score_ablation_no_interactions,
    "Ablation 2 (Global Median Age Imputation vs Grouped Median)": score_ablation_global_imputation,
    "Ablation 3 (Weighted Blend 0.85 CatBoost / 0.15 NN vs 50/50 Blend)": score_ablation_weighted_blend,
}

print("=" * 75)
print("                    ABLATION STUDY PERFORMANCE REPORT                    ")
print("=" * 75)
for name, score in results.items():
    delta = score - baseline_50_50
    print(f"{name:<68} : {score:.5f} (Delta: {delta:+.5f})")
print("=" * 75)

impacts = {
    "Interaction Features": abs(score_ablation_no_interactions - baseline_50_50),
    "Grouped vs Global Imputation": abs(
        score_ablation_global_imputation - baseline_50_50
    ),
    "Ensemble Blending Weights": abs(score_ablation_weighted_blend - baseline_50_50),
}
most_impactful = max(impacts, key=impacts.get)

print(
    f"Conclusion: '{most_impactful}' contributes the most to the overall performance variation with an absolute impact of {impacts[most_impactful]:.5f}."
)