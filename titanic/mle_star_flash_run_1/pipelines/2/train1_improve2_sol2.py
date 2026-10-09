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
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Configurations for diverse CatBoost models
cb_configs = [
    {"iterations": 550, "learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 3.0, "random_seed": 42},
    {"iterations": 600, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 5.0, "random_seed": 123},
    {"iterations": 500, "learning_rate": 0.05, "depth": 7, "l2_leaf_reg": 2.0, "random_seed": 789},
]

# Configurations for diverse XGBoost models
xgb_configs = [
    {
        "n_estimators": 350,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    {
        "n_estimators": 400,
        "learning_rate": 0.025,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "random_state": 123,
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.04,
        "max_depth": 3,
        "subsample": 0.75,
        "colsample_bytree": 0.85,
        "random_state": 789,
    },
]

cb_val_probs = []
for config in cb_configs:
    cb_model = CatBoostClassifier(
        **config,
        cat_features=cat_features,
        eval_metric="Accuracy",
        verbose=0,
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    cb_val_probs.append(cb_model.predict_proba(X_val)[:, 1])

xgb_val_probs = []
for config in xgb_configs:
    xgb_model = XGBClassifier(
        **config,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)
    xgb_val_probs.append(xgb_model.predict_proba(X_val)[:, 1])

# Aggregate predictions with CatBoost-prioritized weighted soft-voting
cb_mean_probs = np.mean(cb_val_probs, axis=0)
xgb_mean_probs = np.mean(xgb_val_probs, axis=0)
ensemble_val_probs = 0.65 * cb_mean_probs + 0.35 * xgb_mean_probs

# Apply natural 0.5 decision boundary
val_preds = (ensemble_val_probs >= 0.5).astype(int)

final_validation_score = accuracy_score(y_val, val_preds)

print(f"Final Validation Performance: {final_validation_score}")