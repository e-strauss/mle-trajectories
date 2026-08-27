import json
import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xgboost as xgb


def build_physiological_features(df):
    """Computes domain-specific physiological ratios, composite indices, and missingness signals."""
    df = df.copy()

    # 1. Missingness Indicators
    missing_cols = [
        "bmi",
        "calorie_expenditure",
        "exercise_duration",
        "heart_rate",
        "sleep_duration",
        "step_count",
        "water_intake",
    ]
    for col in missing_cols:
        if col in df.columns:
            df[f"{col}_is_na"] = df[col].isna().astype(int)
    df["total_missing_health_logs"] = df[
        [f"{col}_is_na" for col in missing_cols if f"{col}_is_na" in df.columns]
    ].sum(axis=1)

    # 2. Ordinal Lifestyle Mappings
    stress_map = {"low": 1, "medium": 2, "high": 3}
    sleep_qual_map = {"poor": 1, "average": 2, "good": 3}
    activity_map = {"sedentary": 1, "moderate": 2, "active": 3}
    smoke_alc_map = {"no": 0, "occasional": 1, "yes": 2}
    diet_map = {"veg": 1, "balanced": 2, "non-veg": 3}
    gender_map = {"female": 1, "male": 2, "other": 3}

    df["stress_level_num"] = df["stress_level"].map(stress_map).fillna(0)
    df["sleep_quality_num"] = df["sleep_quality"].map(sleep_qual_map).fillna(0)
    df["physical_activity_num"] = (
        df["physical_activity_level"].map(activity_map).fillna(0)
    )
    df["smoking_alcohol_num"] = df["smoking_alcohol"].map(smoke_alc_map).fillna(0)
    df["diet_type_num"] = df["diet_type"].map(diet_map).fillna(0)
    df["gender_num"] = df["gender"].map(gender_map).fillna(0)

    # 3. Domain Ratios & Interactions
    df["movement_efficiency"] = df["step_count"] / (df["exercise_duration"] + 1.0)
    df["metabolic_burn_rate"] = df["calorie_expenditure"] / (df["bmi"] + 1.0)
    df["calorie_per_step"] = df["calorie_expenditure"] / (df["step_count"] + 1.0)
    df["restorative_sleep_score"] = df["sleep_duration"] * df["sleep_quality_num"]
    df["cardiac_stress_ratio"] = df["heart_rate"] / (df["sleep_duration"] + 1.0)
    df["cardiac_exercise_load"] = df["heart_rate"] * df["exercise_duration"]
    df["hydration_activity_index"] = df["water_intake"] / (
        (df["step_count"] / 1000.0) + 1.0
    )

    # Higher-order physiological interaction terms
    df["metabolic_recovery_index"] = (df["calorie_expenditure"] * df["sleep_duration"]) / (df["heart_rate"] + 1.0)
    df["hydration_density_bmi"] = df["water_intake"] / (df["bmi"] + 1.0)
    df["cardiac_metabolic_stress_product"] = df["heart_rate"] * df["stress_level_num"] * (df["bmi"] / (df["water_intake"] + 1.0))

    # Composite Health Risk Index
    df["lifestyle_risk_score"] = (
        df["stress_level_num"] * 1.5
        + df["smoking_alcohol_num"] * 2.0
        - df["sleep_quality_num"] * 1.0
        - df["physical_activity_num"] * 1.0
    )

    df["metabolic_risk_flag"] = (
        (df["bmi"] > 25.0).astype(int)
        + (df["heart_rate"] > 80.0).astype(int)
        + (df["stress_level_num"] >= 2).astype(int)
    )

    # 4. Multi-group Demographic Aggregations (composite keys)
    df["gender_activity"] = df["gender"].astype(str) + "_" + df["physical_activity_level"].astype(str)
    df["diet_smoke"] = df["diet_type"].astype(str) + "_" + df["smoking_alcohol"].astype(str)

    return df


class ResNetBlock(nn.Module):
    """Residual Block with BatchNorm, Dropout, and SiLU activation for tabular features."""

    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return self.act(out + residual)


class TabularResNet(nn.Module):
    """Deep Tabular ResNet designed for health condition multi-class classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                ResNetBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss to optimize hard and rare class predictions for Balanced Accuracy."""

    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_neural_components(
    input_dim: int,
    num_classes: int = 3,
    class_weights: torch.Tensor = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    model = TabularResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.2,
    )
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
    return model, criterion, optimizer, scheduler


def get_gbdt_models(seed: int = 42):
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )

    cb_model = cb.CatBoostClassifier(
        loss_function="MultiClass",
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=0,
    )

    return {
        "lightgbm": lgb_model,
        "xgboost": xgb_model,
        "catboost": cb_model,
    }


class TabularDataset(torch.utils.data.Dataset):

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def main():
    input_dir = "./input"
    working_dir = "./working"
    submission_dir = "./submission"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # 1. Load Raw Datasets
    train_raw = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test_raw = pd.read_csv(os.path.join(input_dir, "test.csv"))

    target_col = "health_condition"
    target_mapping = {"at-risk": 0, "fit": 1, "unhealthy": 2}
    reverse_target_mapping = {v: k for k, v in target_mapping.items()}

    train_raw[target_col + "_encoded"] = train_raw[target_col].map(target_mapping)

    # Stratified Train/Validation Split (80/20)
    train_df, val_df = train_test_split(
        train_raw,
        test_size=0.20,
        random_state=42,
        stratify=train_raw[target_col + "_encoded"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_raw.copy().reset_index(drop=True)

    # Apply Physiological Feature Engineering
    train_feat = build_physiological_features(train_df)
    val_feat = build_physiological_features(val_df)
    test_feat = build_physiological_features(test_df)

    # Multi-group demographic z-scores fit purely on Train split
    num_targets = ["bmi", "heart_rate", "step_count", "calorie_expenditure", "sleep_duration", "water_intake"]
    for grp_col in ["gender_activity", "diet_smoke"]:
        grp_stats = train_feat.groupby(grp_col)[num_targets].agg(["mean", "std"])
        for col in num_targets:
            mean_map = grp_stats[(col, "mean")].to_dict()
            std_map = grp_stats[(col, "std")].to_dict()
            for df in [train_feat, val_feat, test_feat]:
                m = df[grp_col].map(mean_map)
                s = df[grp_col].map(std_map)
                df[f"{col}_{grp_col}_zscore"] = (df[col] - m) / (s + 1e-5)

    # Group Aggregations based purely on Train split
    group_cols = ["physical_activity_level"]
    agg_targets = ["bmi", "heart_rate", "step_count", "calorie_expenditure"]

    group_stats = (
        train_feat.groupby(group_cols)[agg_targets].agg(["mean", "std"]).reset_index()
    )

    group_stats.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0] for c in group_stats.columns
    ]

    for target in agg_targets:
        mean_map = group_stats.set_index(group_cols[0])[f"{target}_mean"].to_dict()
        std_map = group_stats.set_index(group_cols[0])[f"{target}_std"].to_dict()
        for df in [train_feat, val_feat, test_feat]:
            m = df[group_cols[0]].map(mean_map)
            s = df[group_cols[0]].map(std_map)
            df[f"{target}_activity_zscore"] = (df[target] - m) / (s + 1e-5)

    # Target Encoding using 5-Fold OOF on Training Set
    cat_cols_to_encode = [
        "diet_type",
        "physical_activity_level",
        "smoking_alcohol",
        "stress_level",
    ]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for col in cat_cols_to_encode:
        for class_idx in range(3):
            train_feat[f"{col}_te_class_{class_idx}"] = np.nan
            val_feat[f"{col}_te_class_{class_idx}"] = np.nan
            test_feat[f"{col}_te_class_{class_idx}"] = np.nan

        # OOF Target Encoding for Train
        for tr_idx, va_idx in skf.split(
            train_feat, train_feat[target_col + "_encoded"]
        ):
            tr_fold = train_feat.iloc[tr_idx]
            for class_idx in range(3):
                class_counts_col = (
                    tr_fold[tr_fold[target_col + "_encoded"] == class_idx]
                    .groupby(col)[target_col + "_encoded"]
                    .count()
                )
                total_counts_col = tr_fold.groupby(col)[target_col + "_encoded"].count()
                class_means = (class_counts_col / total_counts_col).to_dict()
                mapped_vals = (
                    train_feat.iloc[va_idx][col].map(class_means).fillna(1.0 / 3.0)
                )
                train_feat.loc[
                    train_feat.index[va_idx], f"{col}_te_class_{class_idx}"
                ] = mapped_vals

        # Global Target Encoding Mappings computed on full Train for Val & Test
        for class_idx in range(3):
            global_class_means = (
                train_feat[train_feat[target_col + "_encoded"] == class_idx]
                .groupby(col)[target_col + "_encoded"]
                .count()
                / train_feat.groupby(col)[target_col + "_encoded"].count()
            ).to_dict()
            val_feat[f"{col}_te_class_{class_idx}"] = (
                val_feat[col].map(global_class_means).fillna(1.0 / 3.0)
            )
            test_feat[f"{col}_te_class_{class_idx}"] = (
                test_feat[col].map(global_class_means).fillna(1.0 / 3.0)
            )

    # Identify feature columns
    ignore_cols = [
        target_col,
        target_col + "_encoded",
        "id",
        "diet_type",
        "gender",
        "physical_activity_level",
        "sleep_quality",
        "smoking_alcohol",
        "stress_level",
        "gender_activity",
        "diet_smoke",
    ]
    feature_cols = [c for c in train_feat.columns if c not in ignore_cols]

    X_train = train_feat[feature_cols].astype(np.float32).values
    y_train = train_feat[target_col + "_encoded"].values
    X_val = val_feat[feature_cols].astype(np.float32).values
    y_val = val_feat[target_col + "_encoded"].values
    X_test = test_feat[feature_cols].astype(np.float32).values

    # Train LightGBM Model
    gbdt_models = get_gbdt_models(seed=42)
    lgb_model = gbdt_models["lightgbm"]
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    probs_lgb_val = lgb_model.predict_proba(X_val)
    probs_lgb_test = lgb_model.predict_proba(X_test)

    # Train XGBoost Model
    xgb_model = gbdt_models["xgboost"]
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    probs_xgb_val = xgb_model.predict_proba(X_val)
    probs_xgb_test = xgb_model.predict_proba(X_test)

    # Train CatBoost Model
    cb_model = gbdt_models["catboost"]
    cb_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=0,
    )
    probs_cb_val = cb_model.predict_proba(X_val)
    probs_cb_test = cb_model.predict_proba(X_test)

    # Train PyTorch TabularResNet Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_nn = scaler.fit_transform(imputer.fit_transform(X_train))
    X_va_nn = scaler.transform(imputer.transform(X_val))
    X_te_nn = scaler.transform(imputer.transform(X_test))

    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(
        len(y_train) / (len(class_counts) * class_counts), dtype=torch.float32
    ).to(device)

    nn_model, criterion, optimizer, scheduler = build_neural_components(
        input_dim=X_tr_nn.shape[1],
        num_classes=3,
        class_weights=class_weights,
        lr=1e-3,
    )
    nn_model = nn_model.to(device)

    train_dataset = TabularDataset(X_tr_nn, y_train)
    val_dataset = TabularDataset(X_va_nn, y_val)
    test_dataset = TabularDataset(X_te_nn)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2048, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4096, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4096, shuffle=False
    )

    epochs = 15
    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(epochs):
        nn_model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bx.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * bx.size(0)
        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = nn_model.state_dict().copy()

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

    if best_state_dict is not None:
        nn_model.load_state_dict(best_state_dict)

    nn_model.eval()
    nn_val_probs = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            out = nn_model(bx)
            nn_val_probs.append(F.softmax(out, dim=1).cpu().numpy())
    probs_nn_val = np.vstack(nn_val_probs)

    nn_test_probs = []
    with torch.no_grad():
        for bx in test_loader:
            bx = bx.to(device)
            out = nn_model(bx)
            nn_test_probs.append(F.softmax(out, dim=1).cpu().numpy())
    probs_nn_test = np.vstack(nn_test_probs)

    # Blending Model Probabilities
    blend_probs_val = (
        0.35 * probs_lgb_val
        + 0.30 * probs_xgb_val
        + 0.25 * probs_cb_val
        + 0.10 * probs_nn_val
    )

    blend_probs_test = (
        0.35 * probs_lgb_test
        + 0.30 * probs_xgb_test
        + 0.25 * probs_cb_test
        + 0.10 * probs_nn_test
    )

    # Class Threshold Optimization via Nelder-Mead for Balanced Accuracy
    def objective(weights):
        w = np.array(weights)
        scaled_probs = blend_probs_val * w
        preds = np.argmax(scaled_probs, axis=1)
        return -balanced_accuracy_score(y_val, preds)

    res = minimize(objective, [1.0, 1.0, 1.0], method="Nelder-Mead")
    best_weights = res.x

    final_val_preds = np.argmax(blend_probs_val * best_weights, axis=1)
    final_val_score = balanced_accuracy_score(y_val, final_val_preds)

    # Generate Test Submission
    final_test_preds = np.argmax(blend_probs_test * best_weights, axis=1)
    test_condition_str = [reverse_target_mapping[p] for p in final_test_preds]

    sub_df = pd.DataFrame(
        {"id": test_df["id"].values, "health_condition": test_condition_str}
    )
    sub_df.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)

    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()