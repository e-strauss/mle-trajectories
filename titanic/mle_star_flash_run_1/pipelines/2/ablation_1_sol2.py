import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load dataset
train_path = "./input/train.csv"
df = pd.read_csv(train_path)

target = "Survived"
y = df[target].copy()


def build_features(drop_group=None):
    X = pd.DataFrame(index=df.index)

    # Base demographic & numerical features
    if drop_group != "base_numerical":
        X["Age"] = df["Age"]
        X["SibSp"] = df["SibSp"]
        X["Parch"] = df["Parch"]
        X["Fare"] = df["Fare"]

    X["Pclass"] = df["Pclass"].astype(str)
    X["Sex"] = df["Sex"].astype(str)
    X["Embarked"] = df["Embarked"].fillna("Missing").astype(str)

    # Group 1: Title and Demographic Interactions
    if drop_group != "title_and_interactions":
        title = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
        surname = df["Name"].str.split(",").str[0].str.strip()
        X["Title"] = title.fillna("Missing").astype(str)
        X["Surname"] = surname.fillna("Missing").astype(str)
        X["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
        X["IsMother"] = (
            (df["Sex"] == "female")
            & (df["Parch"] > 0)
            & (df["Age"] > 18)
            & (title != "Miss")
        ).astype(int).astype(str)

    # Group 2: Ticket, Family Size, and Relative Economic Features
    if drop_group != "ticket_and_economic":
        surname = df["Name"].str.split(",").str[0].str.strip()
        ticket_clean = (
            df["Ticket"].astype(str).str.replace(r"[\./]", "", regex=True).str.strip()
        )
        ticket_prefix = ticket_clean.apply(
            lambda x: x.split()[0] if len(x.split()) > 1 else "None"
        )
        X["TicketPrefix"] = ticket_prefix.astype(str)
        X["TicketFrequency"] = df["Ticket"].map(df["Ticket"].value_counts())
        X["SurnameFrequency"] = surname.map(surname.value_counts())
        X["LogFare"] = np.log1p(
            np.maximum(0, df["Fare"].fillna(df["Fare"].median()))
        )
        median_fare_by_pclass_embarked = df.groupby(["Pclass", "Embarked"])[
            "Fare"
        ].transform("median")
        X["FareRatioByPclass"] = df["Fare"] / (
            median_fare_by_pclass_embarked + 1e-5
        )

    # Ensure categorical types
    for col in X.columns:
        if (
            X[col].dtype == "object"
            or col in ["Pclass", "Sex", "Embarked", "Title", "Surname", "Sex_Pclass", "IsMother", "TicketPrefix"]
        ):
            X[col] = X[col].astype("category")

    return X


def evaluate_pipeline(X, y):
    cat_features = [
        col for col in X.columns if X[col].dtype.name == "category" or X[col].dtype == "object"
    ]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric="Accuracy",
        random_seed=42,
        verbose=0,
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        random_seed=42,
        eval_metric="logloss",
    )

    cb_model.fit(train_pool, eval_set=val_pool, verbose=False)
    xgb_model.fit(X_train, y_train)

    cb_val_probs = cb_model.predict_proba(val_pool)[:, 1]
    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]

    ensemble_val_probs = (cb_val_probs + xgb_val_probs) / 2.0
    val_preds = (ensemble_val_probs >= 0.5).astype(int)

    return accuracy_score(y_val, val_preds)


# 1. Full Engineered Pipeline
X_full = build_features()
score_full = evaluate_pipeline(X_full, y)
print(f"Full Feature Pipeline Validation Accuracy: {score_full:.5f}")

# 2. Ablation A: Drop Title & Demographic Interactions
X_no_title = build_features(drop_group="title_and_interactions")
score_no_title = evaluate_pipeline(X_no_title, y)
delta_no_title = score_no_title - score_full
print(
    f"Ablation 1 (Drop Title & Demographic Interactions) Validation Accuracy: {score_no_title:.5f} (Delta: {delta_no_title:+.5f})"
)

# 3. Ablation B: Drop Ticket, Group Frequencies & Relative Economic Features
X_no_ticket = build_features(drop_group="ticket_and_economic")
score_no_ticket = evaluate_pipeline(X_no_ticket, y)
delta_no_ticket = score_no_ticket - score_full
print(
    f"Ablation 2 (Drop Ticket, Group & Relative Economic Features) Validation Accuracy: {score_no_ticket:.5f} (Delta: {delta_no_ticket:+.5f})"
)

# 4. Ablation C: Drop Base Numerical Features (Age, Fare, SibSp, Parch)
X_no_numerical = build_features(drop_group="base_numerical")
score_no_numerical = evaluate_pipeline(X_no_numerical, y)
delta_no_numerical = score_no_numerical - score_full
print(
    f"Ablation 3 (Drop Base Numerical Features: Age, Fare, SibSp, Parch) Validation Accuracy: {score_no_numerical:.5f} (Delta: {delta_no_numerical:+.5f})"
)

# Determine the most critical component
ablation_drops = {
    "Title & Demographic Interactions": -delta_no_title,
    "Ticket, Group & Relative Economic Features": -delta_no_ticket,
    "Base Numerical Features (Age, Fare, SibSp, Parch)": -delta_no_numerical,
}

most_critical = max(ablation_drops, key=ablation_drops.get)
print(
    f"\nConclusion: The component contributing most to overall performance is '{most_critical}' with a performance drop of {ablation_drops[most_critical]:.5f} when removed."
)

print(f"Final Validation Performance: {score_full:.5f}")