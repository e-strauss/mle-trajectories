import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load dataset
train_path = "./input/train.csv"
df = pd.read_csv(train_path)

# Feature selection
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

X = df[features].copy()
y = df[target].copy()

# Handle categorical missing values and types
cat_features = ["Sex", "Embarked", "Pclass"]
X["Embarked"] = X["Embarked"].fillna("Missing")
for col in cat_features:
    X[col] = X[col].astype("category")

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Create copies for feature engineering
X_tr = X_train.copy()
X_v = X_val.copy()

# Identify numeric and categorical columns
cat_cols = list(cat_features) if "cat_features" in locals() else [c for c in X_tr.columns if X_tr[c].dtype == "object" or X_tr[c].dtype.name == "category"]
num_cols = [c for c in X_tr.columns if c not in cat_cols and np.issubdtype(X_tr[c].dtype, np.number)]

# 1. Frequency encoding for categorical columns
for col in cat_cols:
    freq_map = X_tr[col].value_counts(normalize=True).to_dict()
    X_tr[f"{col}_freq"] = X_tr[col].map(freq_map).fillna(0).astype(float)
    X_v[f"{col}_freq"] = X_v[col].map(freq_map).fillna(0).astype(float)

# 2. Group aggregation features (mean & std of numeric columns grouped by top categorical columns)
if len(cat_cols) > 0 and len(num_cols) > 0:
    for cat_col in cat_cols[:3]:
        for num_col in num_cols[:4]:
            grouped = X_tr.groupby(cat_col)[num_col].agg(["mean", "std"]).reset_index()
            grouped.columns = [cat_col, f"{num_col}_mean_by_{cat_col}", f"{num_col}_std_by_{cat_col}"]
            
            X_tr = X_tr.merge(grouped, on=cat_col, how="left")
            X_v = X_v.merge(grouped, on=cat_col, how="left")

# 3. Numeric interaction / ratio features
if len(num_cols) >= 2:
    for i in range(min(3, len(num_cols) - 1)):
        c1, c2 = num_cols[i], num_cols[i + 1]
        X_tr[f"{c1}_div_{c2}"] = X_tr[c1] / (X_tr[c2].abs() + 1e-5)
        X_v[f"{c1}_div_{c2}"] = X_v[c1] / (X_v[c2].abs() + 1e-5)
        X_tr[f"{c1}_mult_{c2}"] = X_tr[c1] * X_tr[c2]
        X_v[f"{c1}_mult_{c2}"] = X_v[c1] * X_v[c2]

# Ensure categorical columns have proper 'category' dtype for XGBoost and LightGBM
for col in cat_cols:
    X_tr[col] = X_tr[col].astype("category")
    X_v[col] = pd.Categorical(X_v[col], categories=X_tr[col].cat.categories)

seeds = [42, 2023, 777]
cb_preds = np.zeros(len(X_v))
xgb_preds = np.zeros(len(X_v))
lgb_preds = np.zeros(len(X_v))

for seed in seeds:
    # CatBoost Classifier
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

    # XGBoost Classifier
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

    # LightGBM Classifier
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

# Triad Ensemble (equal probability averaging)
ensemble_val_probs = (cb_preds + xgb_preds + lgb_preds) / 3.0
val_preds = (ensemble_val_probs >= 0.5).astype(int)

final_validation_score = accuracy_score(y_val, val_preds)

print(f"Final Validation Performance: {final_validation_score}")