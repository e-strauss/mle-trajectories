import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load dataset and prepare base split
train_path = "./input/train.csv"
df = pd.read_csv(train_path)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

X = df[features].copy()
y = df[target].copy()

cat_cols = ["Sex", "Embarked", "Pclass"]
X["Embarked"] = X["Embarked"].fillna("Missing")
for col in cat_cols:
    X[col] = X[col].astype("category")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def build_engineered_features(
    X_tr_in,
    X_v_in,
    use_freq=True,
    use_group_agg=True,
    use_interactions=True,
):
    X_tr = X_tr_in.copy()
    X_v = X_v_in.copy()
    num_cols = [
        c
        for c in X_tr.columns
        if c not in cat_cols and np.issubdtype(X_tr[c].dtype, np.number)
    ]

    # 1. Frequency encoding
    if use_freq:
        for col in cat_cols:
            freq_map = X_tr[col].value_counts(normalize=True).to_dict()
            X_tr[f"{col}_freq"] = X_tr[col].map(freq_map).fillna(0).astype(float)
            X_v[f"{col}_freq"] = X_v[col].map(freq_map).fillna(0).astype(float)

    # 2. Group aggregation features
    if use_group_agg and len(cat_cols) > 0 and len(num_cols) > 0:
        for cat_col in cat_cols[:3]:
            for num_col in num_cols[:4]:
                grouped = (
                    X_tr.groupby(cat_col, observed=False)[num_col]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                grouped.columns = [
                    cat_col,
                    f"{num_col}_mean_by_{cat_col}",
                    f"{num_col}_std_by_{cat_col}",
                ]
                X_tr = X_tr.merge(grouped, on=cat_col, how="left")
                X_v = X_v.merge(grouped, on=cat_col, how="left")

    # 3. Numeric interactions
    if use_interactions and len(num_cols) >= 2:
        for i in range(min(3, len(num_cols) - 1)):
            c1, c2 = num_cols[i], num_cols[i + 1]
            X_tr[f"{c1}_div_{c2}"] = X_tr[c1] / (X_tr[c2].abs() + 1e-5)
            X_v[f"{c1}_div_{c2}"] = X_v[c1] / (X_v[c2].abs() + 1e-5)
            X_tr[f"{c1}_mult_{c2}"] = X_tr[c1] * X_tr[c2]
            X_v[f"{c1}_mult_{c2}"] = X_v[c1] * X_v[c2]

    for col in cat_cols:
        X_tr[col] = X_tr[col].astype("category")
        X_v[col] = pd.Categorical(X_v[col], categories=X_tr[col].cat.categories)

    return X_tr, X_v


def train_and_eval(X_tr, X_v, seeds=[42, 2023, 777], use_lgb=True):
    cb_preds = np.zeros(len(X_v))
    xgb_preds = np.zeros(len(X_v))
    lgb_preds = np.zeros(len(X_v))

    for seed in seeds:
        cb_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=3.0,
            subsample=0.8,
            cat_features=cat_cols,
            eval_metric="Logloss",
            random_seed=seed,
            verbose=0,
        )
        cb_model.fit(X_tr, y_train, eval_set=(X_v, y_val), verbose=False)
        cb_preds += cb_model.predict_proba(X_v)[:, 1] / len(seeds)

        xgb_model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.5,
            enable_categorical=True,
            tree_method="hist",
            random_state=seed,
            eval_metric="logloss",
        )
        xgb_model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)
        xgb_preds += xgb_model.predict_proba(X_v)[:, 1] / len(seeds)

        if use_lgb:
            lgb_model = LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=5,
                num_leaves=31,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.5,
                random_state=seed,
                verbose=-1,
            )
            lgb_model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], callbacks=[])
            lgb_preds += lgb_model.predict_proba(X_v)[:, 1] / len(seeds)

    if use_lgb:
        ensemble_val_probs = (cb_preds + xgb_preds + lgb_preds) / 3.0
    else:
        ensemble_val_probs = (cb_preds + xgb_preds) / 2.0

    val_preds = (ensemble_val_probs >= 0.5).astype(int)
    return accuracy_score(y_val, val_preds)


# 1. Baseline Full Triad Pipeline
X_tr_full, X_v_full = build_engineered_features(X_train, X_val)
baseline_score = train_and_eval(X_tr_full, X_v_full, seeds=[42, 2023, 777], use_lgb=True)

# Ablation 1: Disable Multi-Seed Averaging (Single Seed: 42)
ablation_1_score = train_and_eval(X_tr_full, X_v_full, seeds=[42], use_lgb=True)

# Ablation 2: Remove LightGBM from the Ensemble (CatBoost + XGBoost only)
ablation_2_score = train_and_eval(X_tr_full, X_v_full, seeds=[42, 2023, 777], use_lgb=False)

# Ablation 3: Remove Group Aggregations (Keep Frequency & Numeric Interactions)
X_tr_no_agg, X_v_no_agg = build_engineered_features(
    X_train, X_val, use_freq=True, use_group_agg=False, use_interactions=True
)
ablation_3_score = train_and_eval(X_tr_no_agg, X_v_no_agg, seeds=[42, 2023, 777], use_lgb=True)

# Ablation 4: Remove Numeric Interactions (Keep Frequency & Group Aggregations)
X_tr_no_int, X_v_no_int = build_engineered_features(
    X_train, X_val, use_freq=True, use_group_agg=True, use_interactions=False
)
ablation_4_score = train_and_eval(X_tr_no_int, X_v_no_int, seeds=[42, 2023, 777], use_lgb=True)

results = {
    "Full Pipeline (Baseline)": baseline_score,
    "Ablation 1 (Disable Multi-Seed Averaging)": ablation_1_score,
    "Ablation 2 (Drop LightGBM from Triad Ensemble)": ablation_2_score,
    "Ablation 3 (Drop Group Aggregations)": ablation_3_score,
    "Ablation 4 (Drop Numeric Interactions)": ablation_4_score,
}

print("=== Ablation Study Results ===")
for name, score in results.items():
    delta = score - baseline_score
    print(f"{name}: Accuracy = {score:.5f} (Delta = {delta:+.5f})")

# Determine the component whose removal caused the largest decrease in performance
ablation_drops = {
    "Multi-Seed Averaging": baseline_score - ablation_1_score,
    "LightGBM in Triad Ensemble": baseline_score - ablation_2_score,
    "Group Aggregations": baseline_score - ablation_3_score,
    "Numeric Interactions": baseline_score - ablation_4_score,
}

most_impactful_part = max(ablation_drops, key=ablation_drops.get)
print("\nMost Critical Component:")
print(f"'{most_impactful_part}' contributes the most to the overall performance with a drop of {ablation_drops[most_impactful_part]:.5f} when removed/modified.")