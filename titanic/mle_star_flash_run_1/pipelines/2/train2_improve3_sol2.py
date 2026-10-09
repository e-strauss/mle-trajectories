import os
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def engineer_domain_features(df):
    df_feat = df.copy()

    # Title extraction from Name
    if "Name" in df_feat.columns:
        df_feat["Title"] = df_feat["Name"].str.extract(
            r" ([A-Za-z]+)\.", expand=False
        )
        title_mapping = {
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Dr": "Officer",
            "Rev": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Mlle": "Miss",
            "Mme": "Mrs",
            "Ms": "Miss",
            "Capt": "Officer",
            "Lady": "Royalty",
            "Sir": "Royalty",
            "Countess": "Royalty",
            "Jonkheer": "Royalty",
            "Don": "Royalty",
            "Dona": "Royalty",
        }
        df_feat["Title"] = df_feat["Title"].map(title_mapping).fillna("Other")
    else:
        df_feat["Title"] = "Unknown"

    # Family Size and Grouping
    sibsp = df_feat["SibSp"] if "SibSp" in df_feat.columns else 0
    parch = df_feat["Parch"] if "Parch" in df_feat.columns else 0
    df_feat["FamilySize"] = sibsp + parch + 1
    df_feat["IsAlone"] = (df_feat["FamilySize"] == 1).astype(float)
    df_feat["IsSmallFamily"] = (
        (df_feat["FamilySize"] >= 2) & (df_feat["FamilySize"] <= 4)
    ).astype(float)
    df_feat["IsLargeFamily"] = (df_feat["FamilySize"] > 4).astype(float)

    # Fare Transformation
    if "Fare" in df_feat.columns:
        fare = df_feat["Fare"].fillna(df_feat["Fare"].median())
        df_feat["Fare_filled"] = fare
        df_feat["Fare_log"] = np.log1p(np.maximum(0, fare))
        df_feat["FarePerPerson"] = df_feat["Fare_log"] / df_feat["FamilySize"]
    else:
        df_feat["Fare_filled"] = 0.0
        df_feat["Fare_log"] = 0.0
        df_feat["FarePerPerson"] = 0.0

    # Age Imputation by (Title, Pclass) median
    if "Age" in df_feat.columns:
        if "Pclass" in df_feat.columns:
            age_imputed = df_feat.groupby(["Title", "Pclass"])["Age"].transform(
                lambda s: s.fillna(s.median())
            )
        else:
            age_imputed = df_feat["Age"]
        age_clean = age_imputed.fillna(df_feat["Age"].median()).fillna(28.0)
        df_feat["Age_filled"] = age_clean
        df_feat["IsChild"] = (df_feat["Age_filled"] <= 12).astype(float)
        df_feat["Age_Pclass"] = (
            df_feat["Age_filled"]
            * df_feat.get("Pclass", pd.Series(3, index=df.index)).astype(float)
        )
    else:
        df_feat["Age_filled"] = 28.0
        df_feat["IsChild"] = 0.0
        df_feat["Age_Pclass"] = 84.0

    # Cabin Features
    if "Cabin" in df_feat.columns:
        df_feat["Deck"] = (
            df_feat["Cabin"].astype(str).str[0].fillna("M").replace("n", "M")
        )
        df_feat["HasCabin"] = (
            df_feat["Cabin"].notna() & (df_feat["Cabin"] != "")
        ).astype(float)
    else:
        df_feat["Deck"] = "M"
        df_feat["HasCabin"] = 0.0

    # Embarked & Sex
    df_feat["Embarked_clean"] = (
        df_feat["Embarked"].fillna("S").astype(str)
        if "Embarked" in df_feat.columns
        else "S"
    )
    df_feat["Sex_clean"] = (
        df_feat["Sex"].fillna("male").astype(str)
        if "Sex" in df_feat.columns
        else "male"
    )
    df_feat["IsFemale"] = (df_feat["Sex_clean"] == "female").astype(float)
    df_feat["Pclass_num"] = (
        df_feat["Pclass"].astype(float) if "Pclass" in df_feat.columns else 3.0
    )

    return df_feat


def preprocess_data(train_df, test_df=None):
    train_eng = engineer_domain_features(train_df)
    if test_df is not None:
        test_eng = engineer_domain_features(test_df)
        combined = pd.concat([train_eng, test_eng], axis=0, ignore_index=True)
    else:
        combined = train_eng

    # Categorical columns to one-hot encode
    cat_cols = ["Title", "Deck", "Embarked_clean", "Sex_clean"]
    num_cols = [
        "Pclass_num",
        "Age_filled",
        "Fare_filled",
        "Fare_log",
        "FarePerPerson",
        "FamilySize",
        "IsAlone",
        "IsSmallFamily",
        "IsLargeFamily",
        "IsChild",
        "Age_Pclass",
        "HasCabin",
        "IsFemale",
    ]

    dummies = pd.get_dummies(combined[cat_cols], drop_first=True, dtype=float)
    features_df = pd.concat([combined[num_cols], dummies], axis=1)

    X_train = features_df.iloc[: len(train_df)].copy()
    X_test = (
        features_df.iloc[len(train_df) :].copy() if test_df is not None else None
    )

    return X_train, X_test


# Load datasets
train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

y = train_df["Survived"].values
X, X_test = preprocess_data(train_df, test_df)

# Scale features for linear models
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X), columns=X.columns, index=X.index
)
X_test_scaled = (
    pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    if X_test is not None
    else None
)

# Define models
models = {
    "rf": RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "et": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "gb": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    ),
    "lr": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
}

if LGBMClassifier is not None:
    models["lgb"] = LGBMClassifier(
        n_estimators=120,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

if XGBClassifier is not None:
    models["xgb"] = XGBClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

if CatBoostClassifier is not None:
    models["cat"] = CatBoostClassifier(
        iterations=200,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=3.0,
        subsample=0.8,
        random_seed=42,
        verbose=0,
    )

# Stratified K-Fold Out-Of-Fold Evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = {name: np.zeros(len(train_df)) for name in models}
test_preds = (
    {name: np.zeros(len(test_df)) for name in models}
    if test_df is not None
    else {}
)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    y_tr, y_va = y[train_idx], y[val_idx]

    for name, model in models.items():
        if name == "lr":
            X_tr, X_va = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            model.fit(X_tr, y_tr)
            val_probs = model.predict_proba(X_va)[:, 1]
            if test_df is not None:
                test_preds[name] += (
                    model.predict_proba(X_test_scaled)[:, 1] / skf.n_splits
                )
        else:
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            model.fit(X_tr, y_tr)
            val_probs = model.predict_proba(X_va)[:, 1]
            if test_df is not None:
                test_preds[name] += (
                    model.predict_proba(X_test)[:, 1] / skf.n_splits
                )

        oof_preds[name][val_idx] = val_probs

# Meta-learner stacking / ensemble averaging
meta_features = np.column_stack([oof_preds[name] for name in models])
meta_learner = LogisticRegression(C=0.5, random_state=42)
meta_learner.fit(meta_features, y)

oof_ensemble_probs = meta_learner.predict_proba(meta_features)[:, 1]
final_validation_score = accuracy_score(
    y, (oof_ensemble_probs >= 0.5).astype(int)
)

print(f"Final Validation Performance: {final_validation_score}")

# Generate test predictions if test dataset exists
if test_df is not None:
    test_meta_features = np.column_stack([test_preds[name] for name in models])
    final_test_probs = meta_learner.predict_proba(test_meta_features)[:, 1]
    final_test_preds = (final_test_probs >= 0.5).astype(int)

    submission = pd.DataFrame(
        {
            "PassengerId": (
                test_df["PassengerId"]
                if "PassengerId" in test_df.columns
                else range(len(test_df))
            ),
            "Survived": final_test_preds,
        }
    )
    submission.to_csv("submission.csv", index=False)