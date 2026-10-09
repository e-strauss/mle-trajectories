import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

# Load data
train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
has_test = os.path.exists(test_path)
test_df = pd.read_csv(test_path) if has_test else None

y = train_df["Survived"].values

# Combine train and test for consistent feature extraction
if has_test:
    combined_df = pd.concat([train_df.drop(columns=["Survived"]), test_df], ignore_index=True)
else:
    combined_df = train_df.drop(columns=["Survived"]).copy()

# Feature Engineering
# 1. Title Extraction
combined_df["Title"] = combined_df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_mapping = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Royalty",
    "Countess": "Royalty",
    "Capt": "Officer",
    "Col": "Officer",
    "Don": "Royalty",
    "Dr": "Officer",
    "Major": "Officer",
    "Rev": "Officer",
    "Sir": "Royalty",
    "Jonkheer": "Royalty",
    "Dona": "Royalty",
}
combined_df["Title"] = combined_df["Title"].map(title_mapping).fillna("Rare")

# 2. Family Size & Alone Feature
combined_df["FamilySize"] = combined_df["SibSp"] + combined_df["Parch"] + 1
combined_df["IsAlone"] = (combined_df["FamilySize"] == 1).astype(int)

# 3. Deck from Cabin
combined_df["Deck"] = combined_df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) else "U")

# 4. Ticket frequency
ticket_counts = combined_df["Ticket"].value_counts()
combined_df["Ticket_Freq"] = combined_df["Ticket"].map(ticket_counts).fillna(1)

# 5. Adjusted Fare
combined_df["Fare"] = combined_df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.median()))
combined_df["Fare_Per_Person"] = combined_df["Fare"] / combined_df["Ticket_Freq"]

# 6. Age imputation by Title and Pclass
combined_df["Age"] = combined_df.groupby(["Title", "Pclass"])["Age"].transform(
    lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 28.0)
)
combined_df["Age"] = combined_df["Age"].fillna(28.0)

# 7. Embarked fill missing
combined_df["Embarked"] = combined_df["Embarked"].fillna("S")

# 8. Surname
combined_df["Surname"] = combined_df["Name"].apply(
    lambda x: x.split(",")[0].strip() if pd.notna(x) and "," in str(x) else "Unknown"
)
combined_df["Is_Woman_Or_Child"] = (
    (combined_df["Sex"] == "female") | (combined_df["Age"] < 16) | (combined_df["Title"] == "Master")
).astype(int)

# Split back to train and test
n_train = len(train_df)
X_train_full = combined_df.iloc[:n_train].copy()
X_test_full = combined_df.iloc[n_train:].copy() if has_test else None

# Out-of-fold Target Encoding for Family (Woman-Child Group) & Ticket survival
X_train_full["WCG_Survival"] = 0.5
X_train_full["Ticket_Survival"] = 0.5

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for tr_idx, val_idx in skf.split(X_train_full, y):
    tr_data = X_train_full.iloc[tr_idx]
    tr_y = y[tr_idx]

    # Woman-Child surname survival
    wcg_mask = tr_data["Is_Woman_Or_Child"] == 1
    wcg_survival = (
        pd.DataFrame({"Surname": tr_data.loc[wcg_mask, "Surname"], "Survived": tr_y[wcg_mask]})
        .groupby("Surname")["Survived"]
        .mean()
    )

    # Ticket survival
    ticket_survival = (
        pd.DataFrame({"Ticket": tr_data["Ticket"], "Survived": tr_y})
        .groupby("Ticket")["Survived"]
        .mean()
    )

    X_train_full.loc[val_idx, "WCG_Survival"] = X_train_full.loc[val_idx, "Surname"].map(wcg_survival).fillna(0.5)
    X_train_full.loc[val_idx, "Ticket_Survival"] = X_train_full.loc[val_idx, "Ticket"].map(ticket_survival).fillna(0.5)

# For test set, compute full-train based encodings
if has_test:
    wcg_mask = X_train_full["Is_Woman_Or_Child"] == 1
    wcg_survival_full = (
        pd.DataFrame({"Surname": X_train_full.loc[wcg_mask, "Surname"], "Survived": y[wcg_mask]})
        .groupby("Surname")["Survived"]
        .mean()
    )
    ticket_survival_full = (
        pd.DataFrame({"Ticket": X_train_full["Ticket"], "Survived": y})
        .groupby("Ticket")["Survived"]
        .mean()
    )
    X_test_full["WCG_Survival"] = X_test_full["Surname"].map(wcg_survival_full).fillna(0.5)
    X_test_full["Ticket_Survival"] = X_test_full["Ticket"].map(ticket_survival_full).fillna(0.5)

# Select final features
feature_cols = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Fare_Per_Person",
    "Embarked", "Title", "FamilySize", "IsAlone", "Deck", "Ticket_Freq",
    "WCG_Survival", "Ticket_Survival"
]

X_train_fe = X_train_full[feature_cols].copy()
X_test_fe = X_test_full[feature_cols].copy() if has_test else None

cat_cols = ["Sex", "Embarked", "Title", "Deck", "Pclass"]
num_cols = [c for c in feature_cols if c not in cat_cols]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ]
)

# Prepare OOF storage and models
oof_probs = np.zeros(n_train)
test_probs = np.zeros(len(X_test_fe)) if has_test else None

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_fe, y)):
    X_tr_f, X_va_f = X_train_fe.iloc[train_idx], X_train_fe.iloc[val_idx]
    y_tr_f, y_va_f = y[train_idx], y[val_idx]

    X_tr_trans = preprocessor.fit_transform(X_tr_f)
    X_va_trans = preprocessor.transform(X_va_f)
    if has_test:
        X_te_trans = preprocessor.transform(X_test_fe)

    # 1. LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr_trans, y_tr_f)
    p_lgb_val = lgb_model.predict_proba(X_va_trans)[:, 1]

    # 2. CatBoost
    cb_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(X_tr_trans, y_tr_f)
    p_cb_val = cb_model.predict_proba(X_va_trans)[:, 1]

    # 3. XGBoost
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42 + fold,
        eval_metric="logloss"
    )
    xgb_model.fit(X_tr_trans, y_tr_f)
    p_xgb_val = xgb_model.predict_proba(X_va_trans)[:, 1]

    # 4. Logistic Regression
    lr_model = LogisticRegression(penalty="l2", C=0.1, solver="lbfgs", max_iter=1000, random_state=42 + fold)
    lr_model.fit(X_tr_trans, y_tr_f)
    p_lr_val = lr_model.predict_proba(X_va_trans)[:, 1]

    # Fold Ensemble Average
    fold_val_preds = (p_lgb_val + p_cb_val + p_xgb_val + p_lr_val) / 4.0
    oof_probs[val_idx] = fold_val_preds

    if has_test:
        p_lgb_te = lgb_model.predict_proba(X_te_trans)[:, 1]
        p_cb_te = cb_model.predict_proba(X_te_trans)[:, 1]
        p_xgb_te = xgb_model.predict_proba(X_te_trans)[:, 1]
        p_lr_te = lr_model.predict_proba(X_te_trans)[:, 1]
        test_probs += (p_lgb_te + p_cb_te + p_xgb_te + p_lr_te) / (4.0 * n_splits)

# Optimize threshold on OOF
best_threshold = 0.5
best_acc = 0.0

for th in np.linspace(0.35, 0.65, 61):
    acc = accuracy_score(y, (oof_probs >= th).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_threshold = th

final_validation_score = best_acc
print(f"Optimal Threshold: {best_threshold:.4f}")
print(f"Final Validation Performance: {final_validation_score}")

# Generate Submission if test set is present
if has_test and test_probs is not None:
    final_test_preds = (test_probs >= best_threshold).astype(int)
    submission = pd.DataFrame({"Survived": final_test_preds})
    submission.to_csv("submission.csv", index=False)