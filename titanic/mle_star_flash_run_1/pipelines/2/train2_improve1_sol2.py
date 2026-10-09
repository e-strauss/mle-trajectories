import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# 1. Load Data
train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
has_test = os.path.exists(test_path)
test_df = pd.read_csv(test_path) if has_test else None

target = "Survived"
y_arr = train_df[target].values.astype(int)

# 2. Feature Engineering
def engineer_features(df_train, df_test=None):
    if df_test is not None:
        combined = pd.concat([df_train.drop(columns=[target], errors="ignore"), df_test], axis=0, ignore_index=True)
    else:
        combined = df_train.drop(columns=[target], errors="ignore").copy()

    # Title extraction and grouping
    combined["Title"] = combined["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"]
    combined["Title"] = combined["Title"].replace(rare_titles, "Rare")
    combined["Title"] = combined["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    combined["Title"] = combined["Title"].fillna("Missing")

    # Family Size features
    combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
    combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)
    combined["SmallFamily"] = ((combined["FamilySize"] >= 2) & (combined["FamilySize"] <= 4)).astype(int)
    combined["LargeFamily"] = (combined["FamilySize"] > 4).astype(int)

    # Cabin features
    combined["HasCabin"] = combined["Cabin"].notna().astype(int)
    combined["CabinDeck"] = combined["Cabin"].astype(str).str[0]
    combined["CabinDeck"] = combined["CabinDeck"].replace({"n": "Missing", "T": "Missing"})

    # Ticket frequency
    ticket_counts = combined["Ticket"].value_counts()
    combined["TicketGroupSize"] = combined["Ticket"].map(ticket_counts)

    # Imputation for Fare and Age
    combined["Fare"] = combined["Fare"].fillna(combined.groupby("Pclass")["Fare"].transform("median"))
    combined["Age"] = combined["Age"].fillna(combined.groupby(["Pclass", "Title"])["Age"].transform("median"))
    combined["Age"] = combined["Age"].fillna(combined["Age"].median())
    combined["Embarked"] = combined["Embarked"].fillna("S")

    # Interaction / derived features
    combined["FarePerPerson"] = combined["Fare"] / combined["FamilySize"]
    combined["IsMinor"] = (combined["Age"] < 16).astype(int)

    # Encode categorical columns
    cat_cols = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"]
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Drop raw identifier / text columns
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns])

    train_len = len(df_train)
    X_train_proc = combined.iloc[:train_len].reset_index(drop=True)
    X_test_proc = combined.iloc[train_len:].reset_index(drop=True) if df_test is not None else None

    return X_train_proc, X_test_proc

X_processed, X_test_processed = engineer_features(train_df, test_df)

# 3. Stratified K-Fold Ensemble Cross-Validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_processed))
oof_xgb = np.zeros(len(X_processed))
oof_cat = np.zeros(len(X_processed))

if X_test_processed is not None:
    test_preds_lgb = np.zeros(len(X_test_processed))
    test_preds_xgb = np.zeros(len(X_test_processed))
    test_preds_cat = np.zeros(len(X_test_processed))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y_arr)):
    X_tr, y_tr = X_processed.iloc[train_idx], y_arr[train_idx]
    X_va, y_va = X_processed.iloc[val_idx], y_arr[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
    if X_test_processed is not None:
        test_preds_lgb += lgb_model.predict_proba(X_test_processed)[:, 1] / n_splits

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42 + fold
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
    if X_test_processed is not None:
        test_preds_xgb += xgb_model.predict_proba(X_test_processed)[:, 1] / n_splits

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=5,
        subsample=0.8,
        eval_metric="Logloss",
        early_stopping_rounds=50,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        verbose=False
    )
    oof_cat[val_idx] = cat_model.predict_proba(X_va)[:, 1]
    if X_test_processed is not None:
        test_preds_cat += cat_model.predict_proba(X_test_processed)[:, 1] / n_splits

# 4. Optimize Ensemble Weights & Threshold
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat])

def loss_func(params):
    w1, w2, w3, threshold = params
    weights = np.array([w1, w2, w3])
    weights = np.maximum(0, weights)
    if weights.sum() == 0:
        return 0.0
    weights /= weights.sum()
    blend_probs = np.dot(oof_matrix, weights)
    preds = (blend_probs >= threshold).astype(int)
    return -accuracy_score(y_arr, preds)

init_params = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.5]
bounds = [(0, 1), (0, 1), (0, 1), (0.3, 0.7)]
opt_res = minimize(loss_func, init_params, method="Nelder-Mead", bounds=bounds)

opt_weights = np.maximum(0, opt_res.x[:3])
opt_weights /= opt_weights.sum()
opt_threshold = opt_res.x[3]

oof_blend = np.dot(oof_matrix, opt_weights)
oof_final_preds = (oof_blend >= opt_threshold).astype(int)
final_validation_score = accuracy_score(y_arr, oof_final_preds)

print(f"Final Validation Performance: {final_validation_score}")

# 5. Generate Submission if Test Data is Available
if X_test_processed is not None:
    test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cat])
    test_blend = np.dot(test_matrix, opt_weights)
    final_test_preds = (test_blend >= opt_threshold).astype(int)

    submission = pd.DataFrame({"Survived": final_test_preds})
    submission.to_csv("submission.csv", index=False)