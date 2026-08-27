import copy
import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. Feature Engineering & Preprocessing Functions
# ==============================================================================
def load_data(input_dir="./input"):
    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))
    return train_df, test_df


def create_domain_features(df):
    df = df.copy()

    # Ordinal Encodings for Categorical Factors
    sleep_qual_map = {"poor": 0, "average": 1, "good": 2}
    stress_map = {"low": 0, "medium": 1, "high": 2}
    activity_map = {"sedentary": 0, "moderate": 1, "active": 2}
    smok_alc_map = {"no": 0, "occasional": 1, "yes": 2}

    df["sleep_quality_num"] = df["sleep_quality"].map(sleep_qual_map)
    df["stress_level_num"] = df["stress_level"].map(stress_map)
    df["physical_activity_num"] = df["physical_activity_level"].map(activity_map)
    df["smoking_alcohol_num"] = df["smoking_alcohol"].map(smok_alc_map)

    # Missingness Indicators
    num_cols_with_nan = [
        "bmi",
        "calorie_expenditure",
        "exercise_duration",
        "heart_rate",
        "sleep_duration",
        "step_count",
        "water_intake",
    ]
    for col in num_cols_with_nan:
        df[f"{col}_isna"] = df[col].isna().astype(np.float32)

    # Domain Ratios & Interaction Features
    df["cardio_strain"] = (df["heart_rate"] * (df["exercise_duration"] + 1.0)) / (
        df["sleep_duration"] + 1.0
    )
    df["cal_per_step"] = df["calorie_expenditure"] / (df["step_count"] + 1.0)
    df["cal_per_exercise_min"] = df["calorie_expenditure"] / (
        df["exercise_duration"] + 1.0
    )
    df["hydration_to_bmi"] = df["water_intake"] / (df["bmi"] + 1e-5)
    df["sleep_score"] = df["sleep_quality_num"] * df["sleep_duration"]
    df["steps_per_exercise_min"] = df["step_count"] / (df["exercise_duration"] + 1.0)

    # Clinical BMI Bins
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[-np.inf, 18.5, 24.9, 29.9, np.inf], labels=[0, 1, 2, 3]
    ).astype(float)

    # Non-linear transformations
    df["log_step_count"] = np.log1p(np.maximum(0, df["step_count"].fillna(0)))
    df["log_calorie_expenditure"] = np.log1p(
        np.maximum(0, df["calorie_expenditure"].fillna(0))
    )

    return df


def apply_group_aggregations(train_df, test_df):
    group_cols = ["gender", "physical_activity_level"]
    agg_targets = ["heart_rate", "bmi", "calorie_expenditure"]

    train_df = train_df.copy()
    test_df = test_df.copy()

    for target in agg_targets:
        train_df[f"{target}_grp_mean"] = train_df.groupby(group_cols)[target].transform("mean")
        train_df[f"{target}_grp_std"] = train_df.groupby(group_cols)[target].transform("std")

        mean_lookup = train_df.groupby(group_cols)[target].mean().to_dict()
        std_lookup = train_df.groupby(group_cols)[target].std().to_dict()

        test_tuples = pd.Series(list(zip(test_df[group_cols[0]], test_df[group_cols[1]])))
        test_df[f"{target}_grp_mean"] = test_tuples.map(mean_lookup).values
        test_df[f"{target}_grp_std"] = test_tuples.map(std_lookup).values

        train_df[f"{target}_diff_grp_mean"] = (
            train_df[target] - train_df[f"{target}_grp_mean"]
        )
        test_df[f"{target}_diff_grp_mean"] = (
            test_df[target] - test_df[f"{target}_grp_mean"]
        )

    return train_df, test_df


def preprocess_and_encode(train_df, test_df):
    cat_cols = [
        "gender",
        "diet_type",
        "physical_activity_level",
        "sleep_quality",
        "stress_level",
        "smoking_alcohol",
    ]

    for col in cat_cols:
        train_df[col] = train_df[col].fillna("missing").astype(str)
        test_df[col] = test_df[col].fillna("missing").astype(str)

        freq = train_df[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_freq"] = train_df[col].map(freq).fillna(0).astype(np.float32)
        test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(0).astype(np.float32)

        le = LabelEncoder()
        train_df[f"{col}_encoded"] = le.fit_transform(train_df[col])
        test_df[f"{col}_encoded"] = test_df[col].map(
            lambda s: le.transform([s])[0] if s in le.classes_ else -1
        )

    target_le = LabelEncoder()
    train_df["target_encoded"] = target_le.fit_transform(train_df["health_condition"])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["target_encoded"])
    ):
        train_df.loc[val_idx, "fold"] = fold

    return train_df, test_df, target_le


# ==============================================================================
# 2. PyTorch Model Architecture & Focal Loss
# ==============================================================================
class FeatureGatingBlock(nn.Module):

    def __init__(self, num_features):
        super(FeatureGatingBlock, self).__init__()
        self.gate = nn.Sequential(nn.Linear(num_features, num_features), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class GatedResidualBlock(nn.Module):

    def __init__(self, in_features, hidden_features, dropout_rate=0.2):
        super(GatedResidualBlock, self).__init__()
        self.norm1 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = nn.Mish()
        self.drop1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.BatchNorm1d(hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act2 = nn.Mish()
        self.drop2 = nn.Dropout(dropout_rate)

        self.gate = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.drop1(self.act1(self.fc1(out)))
        gate_weights = torch.sigmoid(self.gate(out))
        out = self.norm2(out)
        out = self.drop2(self.act2(self.fc2(out)))
        return residual + (out * gate_weights)


class TabularGatedResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, num_classes=3, dropout_rate=0.2):
        super(TabularGatedResNet, self).__init__()
        self.input_gate = FeatureGatingBlock(input_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )

        self.block1 = GatedResidualBlock(hidden_dim, hidden_dim * 2, dropout_rate)
        self.block2 = GatedResidualBlock(hidden_dim, hidden_dim * 2, dropout_rate)
        self.block3 = GatedResidualBlock(hidden_dim, hidden_dim * 2, dropout_rate)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x_gated = self.input_gate(x)
        h = self.input_layer(x_gated)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        return self.classifier(h)


class FocalLossMultiClass(nn.Module):

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLossMultiClass, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
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
        else:
            return focal_loss


class BalancedAccuracyThresholdOptimizer:

    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.best_weights = np.ones(num_classes)

    def fit(self, y_probs, y_true):
        def loss_func(weights):
            scaled_probs = y_probs * weights
            preds = np.argmax(scaled_probs, axis=1)
            return -balanced_accuracy_score(y_true, preds)

        initial_weights = np.ones(self.num_classes)
        res = minimize(
            loss_func, initial_weights, method="Nelder-Mead", options={"maxiter": 500}
        )
        self.best_weights = res.x
        return self

    def transform(self, y_probs):
        scaled_probs = y_probs * self.best_weights
        return np.argmax(scaled_probs, axis=1)


# ==============================================================================
# 3. Main Execution & Training Pipeline
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and process data
    train_df, test_df = load_data()
    train_df = create_domain_features(train_df)
    test_df = create_domain_features(test_df)
    train_df, test_df = apply_group_aggregations(train_df, test_df)
    train_df, test_df, target_le = preprocess_and_encode(train_df, test_df)

    feature_cols = [
        c
        for c in train_df.columns
        if c not in ["id", "health_condition", "target_encoded", "fold"]
        and train_df[c].dtype != "object"
    ]

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df["target_encoded"].values.astype(np.int64)
    X_test = test_df[feature_cols].values.astype(np.float32)
    folds = train_df["fold"].values
    num_classes = len(np.unique(y))

    n_samples = len(train_df)
    n_test = len(test_df)

    oof_lgb = np.zeros((n_samples, num_classes))
    oof_xgb = np.zeros((n_samples, num_classes))
    oof_cb = np.zeros((n_samples, num_classes))
    oof_nn = np.zeros((n_samples, num_classes))

    test_lgb = np.zeros((n_test, num_classes))
    test_xgb = np.zeros((n_test, num_classes))
    test_cb = np.zeros((n_test, num_classes))
    test_nn = np.zeros((n_test, num_classes))

    lgbm_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "class_weight": "balanced",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    catboost_params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "auto_class_weights": "Balanced",
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "iterations": 200,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }

    for fold in range(5):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. LightGBM
        model_lgb = lgb.LGBMClassifier(**lgbm_params)
        model_lgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        oof_lgb[val_idx] = model_lgb.predict_proba(X_val)
        test_lgb += model_lgb.predict_proba(X_test) / 5.0

        # 2. XGBoost
        model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=20)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_val)
        test_xgb += model_xgb.predict_proba(X_test) / 5.0

        # 3. CatBoost
        model_cb = cb.CatBoostClassifier(**catboost_params)
        model_cb.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=20,
            verbose=False,
        )
        oof_cb[val_idx] = model_cb.predict_proba(X_val)
        test_cb += model_cb.predict_proba(X_test) / 5.0

        # 4. PyTorch TabularGatedResNet
        col_medians = np.nanmedian(X_train, axis=0)
        col_medians[np.isnan(col_medians)] = 0.0

        X_train_imp = np.where(np.isnan(X_train), col_medians, X_train)
        X_val_imp = np.where(np.isnan(X_val), col_medians, X_val)
        X_test_imp = np.where(np.isnan(X_test), col_medians, X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))

        train_loader = DataLoader(
            train_dataset, batch_size=8192, shuffle=True, drop_last=False, num_workers=2
        )
        val_loader = DataLoader(val_dataset, batch_size=16384, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=16384, shuffle=False, num_workers=2)

        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        alpha_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

        model_nn = TabularGatedResNet(
            input_dim=X.shape[1], hidden_dim=256, num_classes=num_classes
        ).to(device)
        criterion = FocalLossMultiClass(gamma=2.0, alpha=alpha_tensor)
        optimizer = torch.optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=6, eta_min=1e-5
        )

        best_val_loss = float("inf")
        best_nn_probs = None
        best_model_state = None

        for epoch in range(6):
            model_nn.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model_nn(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model_nn.eval()
            val_loss = 0.0
            val_probs = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model_nn(bx)
                    loss = criterion(out, by)
                    val_loss += loss.item() * len(by)
                    probs = F.softmax(out, dim=1).cpu().numpy()
                    val_probs.append(probs)

            val_loss /= len(val_dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_probs = np.vstack(val_probs)
                best_model_state = copy.deepcopy(model_nn.state_dict())

        oof_nn[val_idx] = best_nn_probs

        model_nn.load_state_dict(best_model_state)
        model_nn.eval()
        fold_test_probs = []
        with torch.no_grad():
            for (bx,) in test_loader:
                bx = bx.to(device)
                out = model_nn(bx)
                probs = F.softmax(out, dim=1).cpu().numpy()
                fold_test_probs.append(probs)
        test_nn += np.vstack(fold_test_probs) / 5.0

    # Optimize Blending Weights
    def blend_loss(weights):
        w = np.maximum(0, weights)
        w = w / (np.sum(w) + 1e-8)
        blend = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cb + w[3] * oof_nn
        return -balanced_accuracy_score(y, np.argmax(blend, axis=1))

    res = minimize(
        blend_loss,
        [0.35, 0.35, 0.20, 0.10],
        method="Nelder-Mead",
        options={"maxiter": 200},
    )
    opt_w = np.maximum(0, res.x)
    opt_w = opt_w / np.sum(opt_w)

    oof_blend = (
        opt_w[0] * oof_lgb + opt_w[1] * oof_xgb + opt_w[2] * oof_cb + opt_w[3] * oof_nn
    )
    test_blend = (
        opt_w[0] * test_lgb
        + opt_w[1] * test_xgb
        + opt_w[2] * test_cb
        + opt_w[3] * test_nn
    )

    # Threshold Optimization
    threshold_opt = BalancedAccuracyThresholdOptimizer(num_classes=num_classes)
    threshold_opt.fit(oof_blend, y)

    final_oof_preds = threshold_opt.transform(oof_blend)
    final_test_preds = threshold_opt.transform(test_blend)

    # Export Submission
    os.makedirs("./submission", exist_ok=True)
    sub = pd.DataFrame(
        {
            "id": test_df["id"],
            "health_condition": target_le.inverse_transform(final_test_preds),
        }
    )
    sub.to_csv("./submission/submission.csv", index=False)

    score = balanced_accuracy_score(y, final_oof_preds)
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()