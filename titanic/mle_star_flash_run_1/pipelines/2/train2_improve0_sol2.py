import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Load dataset
train_path = "./input/train.csv"
test_path = "./input/test.csv"
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

if os.path.exists(test_path):
    df_test = pd.read_csv(test_path)
    X_test = df_test[features].copy()
    X_test["Embarked"] = X_test["Embarked"].fillna("Missing")
    for col in cat_features:
        X_test[col] = pd.Categorical(X_test[col], categories=X[col].cat.categories)
else:
    df_test = None
    X_test = None

# 5-fold Stratified K-Fold cross-validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cb_probs = np.zeros(len(df))
oof_xgb_probs = np.zeros(len(df))
test_cb_probs = np.zeros(len(df_test)) if df_test is not None else None
test_xgb_probs = np.zeros(len(df_test)) if df_test is not None else None

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

    # Initialize CatBoostClassifier
    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        eval_metric="Accuracy",
        random_seed=42,
        verbose=0,
    )

    # Initialize XGBClassifier
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
    )

    # Train the models
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    xgb_model.fit(X_train, y_train)

    # Predict probabilities on validation fold
    oof_cb_probs[val_idx] = cb_model.predict_proba(X_val)[:, 1]
    oof_xgb_probs[val_idx] = xgb_model.predict_proba(X_val)[:, 1]

    # Predict probabilities on test set
    if df_test is not None:
        test_cb_probs += cb_model.predict_proba(X_test)[:, 1] / n_splits
        test_xgb_probs += xgb_model.predict_proba(X_test)[:, 1] / n_splits

# Ensemble predictions (simple average)
ensemble_val_probs = (oof_cb_probs + oof_xgb_probs) / 2.0
val_preds = (ensemble_val_probs >= 0.5).astype(int)

final_validation_score = accuracy_score(y, val_preds)

print(f"Final Validation Performance: {final_validation_score}")

if df_test is not None:
    ensemble_test_probs = (test_cb_probs + test_xgb_probs) / 2.0
    test_preds = (ensemble_test_probs >= 0.5).astype(int)
    submission = pd.DataFrame({"Survived": test_preds})
    submission.to_csv("submission.csv", index=False)