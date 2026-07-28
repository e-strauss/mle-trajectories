import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.csv"
test_path = "./input/test.csv"

df_train_raw = pd.read_csv(train_path)
df_test_raw = pd.read_csv(test_path)

# Map Target
target_col = "health_condition"
target_map = {"fit": 0, "at-risk": 1, "unhealthy": 2}
inv_target_map = {0: "fit", 1: "at-risk", 2: "unhealthy"}

df_train_raw["target"] = df_train_raw[target_col].map(target_map)
y_all = df_train_raw["target"].values
df_train_raw = df_train_raw.drop(columns=[target_col])

# Leak-Free Stratified 80/20 Split First
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, val_idx = next(sss.split(df_train_raw, y_all))

df_train = df_train_raw.iloc[train_idx].copy().reset_index(drop=True)
y_train = y_all[train_idx]

df_val = df_train_raw.iloc[val_idx].copy().reset_index(drop=True)
y_val = y_all[val_idx]

df_test = df_test_raw.copy().reset_index(drop=True)

# 2. Domain Categorical & Ordinal Mappings
ordinal_maps = {
    "sleep_quality": {"poor": 0, "average": 1, "good": 2},
    "stress_level": {"low": 0, "medium": 1, "high": 2},
    "physical_activity_level": {"sedentary": 0, "moderate": 1, "active": 2},
    "smoking_alcohol": {"no": 0, "occasional": 1, "yes": 2},
    "diet_type": {"veg": 0, "balanced": 1, "non-veg": 2},
    "gender": {"female": 0, "male": 1, "other": 2},
}


def apply_feature_engineering(df):
    data = df.copy()

    # Missingness Indicators
    for col in [
        "sleep_duration",
        "calorie_expenditure",
        "water_intake",
        "bmi",
        "heart_rate",
        "step_count",
    ]:
        data[f"{col}_isna"] = data[col].isna().astype(int)
    data["missing_count_total"] = data.isna().sum(axis=1)

    # Ordinal Encodings
    for col, mapping in ordinal_maps.items():
        data[f"{col}_ord"] = data[col].map(mapping)

    # Domain Physiological Risk Indices
    sq = data["sleep_quality_ord"].fillna(1.0)
    sl = data["stress_level_ord"].fillna(1.0)
    pa = data["physical_activity_level_ord"].fillna(1.0)
    sa = data["smoking_alcohol_ord"].fillna(1.0)
    dt = data["diet_type_ord"].fillna(1.0)

    # Composite Lifestyle Health Risk Factor Index
    data["composite_lifestyle_risk"] = sl + (2.0 - sq) + sa + (2.0 - pa) + (dt * 0.5)

    # Physiological Non-linear Ratios
    data["calories_per_step"] = data["calorie_expenditure"] / (data["step_count"] + 1.0)
    data["calories_per_ex_min"] = data["calorie_expenditure"] / (
        data["exercise_duration"] + 1.0
    )
    data["estimated_bmr_proxy"] = data["calorie_expenditure"] - (
        data["step_count"] * 0.04 + data["exercise_duration"] * 5.0
    )

    data["hr_bmi_ratio"] = data["heart_rate"] / (data["bmi"] + 1e-5)
    data["hr_activity_ratio"] = data["heart_rate"] / (pa + 1.0)
    data["cardio_strain_index"] = (data["heart_rate"] * (1.0 + 0.15 * sl)) / (
        data["sleep_duration"].fillna(7.0) + 0.1
    )

    data["hydration_per_bmi"] = data["water_intake"] / (data["bmi"] + 1e-5)
    data["hydration_per_1k_steps"] = data["water_intake"] / (
        (data["step_count"] / 1000.0) + 1e-5
    )
    data["sleep_efficiency_index"] = (data["sleep_duration"] * (sq + 1.0)) / (sl + 1.0)

    # Non-linear WHO Body Mass Index Dev
    data["bmi_dev_from_optimal"] = np.abs(data["bmi"] - 21.75)
    data["hr_dev_from_normal"] = np.abs(data["heart_rate"] - 70.0)

    return data


df_train_fe = apply_feature_engineering(df_train)
df_val_fe = apply_feature_engineering(df_val)
df_test_fe = apply_feature_engineering(df_test)

# 3. Group Statistics (Computed Strictly on Train set)
group_cols = ["physical_activity_level", "stress_level"]
agg_cols = ["heart_rate", "calorie_expenditure", "step_count", "bmi"]

group_stats = (
    df_train_fe.groupby(group_cols)[agg_cols].agg(["mean", "std"]).reset_index()
)
group_stats.columns = [
    f"{col}_{stat}" if stat else col for col, stat in group_stats.columns
]


def merge_group_stats(df, stats_df):
    merged = pd.merge(df, stats_df, on=group_cols, how="left")
    for col in agg_cols:
        merged[f"{col}_diff_grp_mean"] = merged[col] - merged[f"{col}_mean"]
        merged[f"{col}_ratio_grp_mean"] = merged[col] / (merged[f"{col}_mean"] + 1e-5)
    return merged


df_train_fe = merge_group_stats(df_train_fe, group_stats)
df_val_fe = merge_group_stats(df_val_fe, group_stats)
df_test_fe = merge_group_stats(df_test_fe, group_stats)

# Select numeric features for model training & clustering
drop_cols = [
    "id",
    "target",
    "gender",
    "diet_type",
    "physical_activity_level",
    "sleep_quality",
    "smoking_alcohol",
    "stress_level",
]
feature_cols = [c for c in df_train_fe.columns if c not in drop_cols]

X_train_num = df_train_fe[feature_cols].values
X_val_num = df_val_fe[feature_cols].values
X_test_num = df_test_fe[feature_cols].values

# 4. Fit Imputer, Scaler, and KMeans Clustering strictly on Training Data
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_num)
X_val_imp = imputer.transform(X_val_num)
X_test_imp = imputer.transform(X_test_num)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
train_cluster_dists = kmeans.fit_transform(X_train_scaled)
val_cluster_dists = kmeans.transform(X_val_scaled)
test_cluster_dists = kmeans.transform(X_test_scaled)

cluster_col_names = [f"cluster_dist_{i}" for i in range(6)]

df_train_proc = pd.DataFrame(X_train_imp, columns=feature_cols)
df_val_proc = pd.DataFrame(X_val_imp, columns=feature_cols)
df_test_proc = pd.DataFrame(X_test_imp, columns=feature_cols)

for i, col in enumerate(cluster_col_names):
    df_train_proc[col] = train_cluster_dists[:, i]
    df_val_proc[col] = val_cluster_dists[:, i]
    df_test_proc[col] = test_cluster_dists[:, i]

df_train_proc["target"] = y_train
df_val_proc["target"] = y_val

os.makedirs("./working", exist_ok=True)
df_train_proc.to_parquet("./working/processed_train.parquet", index=False)
df_val_proc.to_parquet("./working/processed_val.parquet", index=False)
df_test_proc.to_parquet("./working/processed_test.parquet", index=False)

# 5. Prepare Arrays for Models
X_train = df_train_proc.drop(columns=["target"]).values
y_train = y_train.astype(int)

X_val = df_val_proc.drop(columns=["target"]).values
y_val = y_val.astype(int)

X_test = df_test_proc.values

input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train))


# 6. Define PyTorch TabGatingNet Model
class TabGatingNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_classes=3, dropout=0.15):
        super(TabGatingNet, self).__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.fc_in = nn.Linear(input_dim, hidden_dim)

        self.gate_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.gate_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc_out1 = nn.Linear(hidden_dim, 64)
        self.fc_out2 = nn.Linear(64, num_classes)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.input_bn(x)
        h = self.act(self.fc_in(x))

        g = torch.sigmoid(self.gate_fc2(h))
        u = self.act(self.gate_fc1(h))
        h_gated = self.bn1(h + g * u)
        h_gated = self.dropout1(h_gated)

        h_seq = h_gated.unsqueeze(1)
        attn_out, _ = self.attn(h_seq, h_seq, h_seq)
        h_attn = self.bn2(h_gated + attn_out.squeeze(1))

        out = self.act(self.fc_out1(h_attn))
        out = self.dropout2(out)
        logits = self.fc_out2(out)
        return logits


# 7. Train PyTorch TabGatingNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=2)

model_nn = TabGatingNet(input_dim=input_dim, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
optimizer = AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=12)

for epoch in range(12):
    model_nn.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model_nn(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    scheduler.step()

model_nn.eval()
nn_val_probs = []
with torch.no_grad():
    for (batch_x,) in val_loader:
        batch_x = batch_x.to(device)
        probs = F.softmax(model_nn(batch_x), dim=1)
        nn_val_probs.append(probs.cpu().numpy())
nn_val_probs = np.vstack(nn_val_probs)

nn_test_probs = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device)
        probs = F.softmax(model_nn(batch_x), dim=1)
        nn_test_probs.append(probs.cpu().numpy())
nn_test_probs = np.vstack(nn_test_probs)

# 8. Train Boosted Trees
clf_lgb = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=num_classes,
    n_estimators=350,
    learning_rate=0.06,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
clf_lgb.fit(X_train, y_train)
lgb_val_probs = clf_lgb.predict_proba(X_val)
lgb_test_probs = clf_lgb.predict_proba(X_test)

clf_cb = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=350,
    learning_rate=0.06,
    depth=6,
    random_seed=42,
    verbose=0,
    thread_count=-1,
)
clf_cb.fit(X_train, y_train)
cb_val_probs = clf_cb.predict_proba(X_val)
cb_test_probs = clf_cb.predict_proba(X_test)

clf_xgb = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    n_estimators=350,
    learning_rate=0.06,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss",
)
clf_xgb.fit(X_train, y_train)
xgb_val_probs = clf_xgb.predict_proba(X_val)
xgb_test_probs = clf_xgb.predict_proba(X_test)

# 9. Joint Multi-Model Blending & Nelder-Mead Balanced Accuracy Threshold Optimization
val_probs_dict = {
    "lgb": lgb_val_probs,
    "cb": cb_val_probs,
    "xgb": xgb_val_probs,
    "nn": nn_val_probs,
}
test_probs_dict = {
    "lgb": lgb_test_probs,
    "cb": cb_test_probs,
    "xgb": xgb_test_probs,
    "nn": nn_test_probs,
}
model_keys = list(val_probs_dict.keys())


def objective(params):
    w = params[: len(model_keys)]
    m = params[len(model_keys) :]

    w_norm = np.exp(w) / np.sum(np.exp(w))
    blended = sum(
        w_norm[i] * val_probs_dict[model_keys[i]] for i in range(len(model_keys))
    )
    adjusted = blended * m
    preds = np.argmax(adjusted, axis=1)
    return -balanced_accuracy_score(y_val, preds)


init_params = np.array([1.0] * len(model_keys) + [1.0, 1.0, 1.0])
opt_res = minimize(
    objective, init_params, method="Nelder-Mead", options={"maxiter": 600}
)

best_w = np.exp(opt_res.x[: len(model_keys)]) / np.sum(
    np.exp(opt_res.x[: len(model_keys)])
)
best_m = opt_res.x[len(model_keys) :]

blended_val = sum(
    best_w[i] * val_probs_dict[model_keys[i]] for i in range(len(model_keys))
)
final_val_preds = np.argmax(blended_val * best_m, axis=1)

blended_test = sum(
    best_w[i] * test_probs_dict[model_keys[i]] for i in range(len(model_keys))
)
final_test_preds = np.argmax(blended_test * best_m, axis=1)

# 10. Output Submission
test_labels = [inv_target_map[p] for p in final_test_preds]
sub_df = pd.DataFrame({"id": df_test_raw["id"], "health_condition": test_labels})

os.makedirs("./submission", exist_ok=True)
sub_df.to_csv("./submission/submission.csv", index=False)

# 11. Print Final Validation Score
score = balanced_accuracy_score(y_val, final_val_preds)
print(f"Final Validation Score: {score}")