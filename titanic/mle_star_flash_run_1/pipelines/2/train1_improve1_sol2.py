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
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Set up Stratified K-Fold cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

cb_oof_probs = np.zeros(len(X_train))
xgb_oof_probs = np.zeros(len(X_train))
cb_val_probs = np.zeros(len(X_val))
xgb_val_probs = np.zeros(len(X_val))

for fold, (train_idx, val_fold_idx) in enumerate(skf.split(X_train, y_train)):
    if hasattr(X_train, "iloc"):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_fold_idx], y_train.iloc[val_fold_idx]
    else:
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_fold_idx], y_train[val_fold_idx]

    # Initialize CatBoostClassifier with strong L2 regularization
    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=5.0,
        cat_features=cat_features,
        eval_metric="Accuracy",
        random_seed=42 + fold,
        verbose=0,
    )
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )

    cb_oof_probs[val_fold_idx] = cb_model.predict_proba(X_va)[:, 1]
    cb_val_probs += cb_model.predict_proba(X_val)[:, 1] / n_splits

    # Initialize XGBClassifier with strong L1/L2 regularization and subsampling
    xgb_model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=3.0,
        enable_categorical=True,
        tree_method="hist",
        random_state=42 + fold,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    xgb_oof_probs[val_fold_idx] = xgb_model.predict_proba(X_va)[:, 1]
    xgb_val_probs += xgb_model.predict_proba(X_val)[:, 1] / n_splits

# Rank-averaged blending on OOF and validation predictions
cb_oof_rank = rankdata(cb_oof_probs) / len(cb_oof_probs)
xgb_oof_rank = rankdata(xgb_oof_probs) / len(xgb_oof_probs)
ensemble_oof_rank = (cb_oof_rank + xgb_oof_rank) / 2.0

cb_val_rank = rankdata(cb_val_probs) / len(cb_val_probs)
xgb_val_rank = rankdata(xgb_val_probs) / len(xgb_val_probs)
ensemble_val_rank = (cb_val_rank + xgb_val_rank) / 2.0

# Probability calibration using Isotonic Regression
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(ensemble_oof_rank, y_train)

calibrated_oof_probs = calibrator.predict(ensemble_oof_rank)
calibrated_val_probs = calibrator.predict(ensemble_val_rank)

# 1D line search to find the optimal classification threshold on OOF calibrated probabilities
best_threshold = 0.5
best_oof_score = 0.0
thresholds = np.linspace(0.1, 0.9, 801)

for th in thresholds:
    score = accuracy_score(y_train, (calibrated_oof_probs >= th).astype(int))
    if score > best_oof_score:
        best_oof_score = score
        best_threshold = th

# Evaluate on hold-out validation set using the optimized threshold
val_preds = (calibrated_val_probs >= best_threshold).astype(int)
final_validation_score = accuracy_score(y_val, val_preds)

print(f"Optimal OOF Threshold: {best_threshold:.4f} (OOF Score: {best_oof_score:.5f})")
print(f"Final Validation Performance: {final_validation_score}")