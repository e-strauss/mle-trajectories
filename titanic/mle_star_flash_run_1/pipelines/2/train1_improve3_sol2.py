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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Ensure categorical columns are appropriately typed
X_train_df = (
    X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
)
X_val_df = X_val.copy() if isinstance(X_val, pd.DataFrame) else pd.DataFrame(X_val)
y_train_arr = np.array(y_train)
y_val_arr = np.array(y_val)

if "cat_features" in locals() and cat_features is not None:
    for col in cat_features:
        if col in X_train_df.columns:
            X_train_df[col] = X_train_df[col].astype("category")
            X_val_df[col] = X_val_df[col].astype("category")
    num_cols = [c for c in X_train_df.columns if c not in cat_features]
    cat_cols = [c for c in X_train_df.columns if c in cat_features]
else:
    cat_cols = list(
        X_train_df.select_dtypes(
            include=["object", "category", "string"]
        ).columns
    )
    num_cols = [c for c in X_train_df.columns if c not in cat_cols]

# Preprocessor for linear / forest models
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop",
)

# Number of folds for Out-Of-Fold stacking
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Models dictionary
models = {
    "catboost": {
        "type": "catboost",
        "params": {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "cat_features": cat_cols if cat_cols else None,
            "eval_metric": "Accuracy",
            "random_seed": 42,
            "verbose": 0,
        },
    },
    "lgbm": {
        "type": "lgbm",
        "params": {
            "n_estimators": 400,
            "learning_rate": 0.04,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        },
    },
    "xgb": {
        "type": "xgb",
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "enable_categorical": True,
            "tree_method": "hist",
            "random_state": 42,
            "eval_metric": "logloss",
        },
    },
    "log_reg": {
        "type": "sklearn_pipe",
        "model": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    },
    "rf": {
        "type": "sklearn_pipe",
        "model": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
    },
}

n_models = len(models)
oof_train_meta = np.zeros((len(X_train_df), n_models))
val_meta = np.zeros((len(X_val_df), n_models))

for m_idx, (m_name, m_cfg) in enumerate(models.items()):
    val_preds_folds = np.zeros((len(X_val_df), n_splits))

    for fold, (train_idx, oof_idx) in enumerate(skf.split(X_train_df, y_train_arr)):
        X_tr, y_tr = X_train_df.iloc[train_idx], y_train_arr[train_idx]
        X_oof, y_oof = X_train_df.iloc[oof_idx], y_train_arr[oof_idx]

        if m_cfg["type"] == "catboost":
            clf = CatBoostClassifier(**m_cfg["params"])
            clf.fit(X_tr, y_tr, verbose=False)
            oof_train_meta[oof_idx, m_idx] = clf.predict_proba(X_oof)[:, 1]
            val_preds_folds[:, fold] = clf.predict_proba(X_val_df)[:, 1]

        elif m_cfg["type"] == "lgbm":
            clf = LGBMClassifier(**m_cfg["params"])
            clf.fit(X_tr, y_tr)
            oof_train_meta[oof_idx, m_idx] = clf.predict_proba(X_oof)[:, 1]
            val_preds_folds[:, fold] = clf.predict_proba(X_val_df)[:, 1]

        elif m_cfg["type"] == "xgb":
            clf = XGBClassifier(**m_cfg["params"])
            clf.fit(X_tr, y_tr)
            oof_train_meta[oof_idx, m_idx] = clf.predict_proba(X_oof)[:, 1]
            val_preds_folds[:, fold] = clf.predict_proba(X_val_df)[:, 1]

        elif m_cfg["type"] == "sklearn_pipe":
            clf = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", m_cfg["model"]),
                ]
            )
            clf.fit(X_tr, y_tr)
            oof_train_meta[oof_idx, m_idx] = clf.predict_proba(X_oof)[:, 1]
            val_preds_folds[:, fold] = clf.predict_proba(X_val_df)[:, 1]

    val_meta[:, m_idx] = val_preds_folds.mean(axis=1)

# Fit regularized meta-learner on out-of-fold predictions
meta_learner = LogisticRegression(C=1.0, penalty="l2", random_state=42)
meta_learner.fit(oof_train_meta, y_train_arr)

# Predict on validation meta-features
ensemble_val_probs = meta_learner.predict_proba(val_meta)[:, 1]
val_preds = (ensemble_val_probs >= 0.5).astype(int)

final_validation_score = accuracy_score(y_val_arr, val_preds)
print(f"Final Validation Performance: {final_validation_score}")