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
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Initialize CatBoostClassifier with early stopping
cb_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=5,
    cat_features=cat_features,
    eval_metric="Accuracy",
    early_stopping_rounds=50,
    random_seed=42,
    verbose=0,
)

# Initialize XGBClassifier with early stopping
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    enable_categorical=True,
    tree_method="hist",
    random_state=42,
    eval_metric="logloss",
)

# Initialize LightGBMClassifier with early stopping
lgb_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Train the models with validation sets for early stopping
cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
xgb_model.fit(
    X_train, y_train, eval_set=[(X_val, y_val)], verbose=False
)

# LightGBM fit (passing categorical features if provided)
if cat_features is not None and len(cat_features) > 0:
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_features,
        callbacks=[],
    )
else:
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])

# Predict probabilities on hold-out validation set
cb_val_probs = cb_model.predict_proba(X_val)[:, 1]
xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
lgb_val_probs = lgb_model.predict_proba(X_val)[:, 1]

# Grid search for optimal blend weights and probability threshold
best_score = 0.0
best_weights = (1 / 3, 1 / 3, 1 / 3)
best_threshold = 0.5

# Search over a grid of weights
for w_cb in np.linspace(0, 1, 11):
    for w_xgb in np.linspace(0, 1 - w_cb, 11):
        w_lgb = 1.0 - w_cb - w_xgb
        if w_lgb < -1e-5:
            continue
        w_lgb = max(0.0, w_lgb)

        blend_probs = (
            w_cb * cb_val_probs + w_xgb * xgb_val_probs + w_lgb * lgb_val_probs
        )

        for thresh in np.linspace(0.35, 0.65, 31):
            preds = (blend_probs >= thresh).astype(int)
            score = accuracy_score(y_val, preds)
            if score > best_score:
                best_score = score
                best_weights = (w_cb, w_xgb, w_lgb)
                best_threshold = thresh

# Compute final ensemble predictions with optimized parameters
ensemble_val_probs = (
    best_weights[0] * cb_val_probs
    + best_weights[1] * xgb_val_probs
    + best_weights[2] * lgb_val_probs
)
val_preds = (ensemble_val_probs >= best_threshold).astype(int)

final_validation_score = accuracy_score(y_val, val_preds)

print(
    f"Optimal Blend Weights (CatBoost, XGBoost, LightGBM): {best_weights}, Optimal Threshold: {best_threshold:.3f}"
)
print(f"Final Validation Performance: {final_validation_score}")