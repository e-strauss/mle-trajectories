import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load dataset
train_path = "./input/train.csv"
df = pd.read_csv(train_path)

import pandas as pd
import numpy as np

target = "Survived"
y = df[target].copy()

# Initialize feature DataFrame from df
X = pd.DataFrame(index=df.index)

# Basic demographic and passenger features
X["Pclass"] = df["Pclass"]
X["Sex"] = df["Sex"]
X["SibSp"] = df["SibSp"]
X["Parch"] = df["Parch"]
X["Fare"] = df["Fare"]
X["Embarked"] = df["Embarked"].fillna("Missing")

# Extract Title for demographic grouping and age imputation
X["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
X["Title"] = X["Title"].fillna("Missing")

# Conditionally impute missing Age by median within (Title, Pclass) group
X["Age"] = df["Age"]
X["Age"] = X.groupby(["Title", "Pclass"])["Age"].transform(lambda s: s.fillna(s.median()))
X["Age"] = X["Age"].fillna(df["Age"].median())

# High-signal demographic indicator
X["WomanOrChild"] = ((X["Sex"] == "female") | (X["Age"] <= 12)).astype(int)

# Ticket-sharing dynamics and adjusted fare
ticket_counts = df["Ticket"].map(df["Ticket"].value_counts())
X["TicketGroupSize"] = ticket_counts
X["AdjustedFare"] = X["Fare"] / X["TicketGroupSize"]

# Cabin count feature
X["CabinCount"] = df["Cabin"].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

# Register categorical features and convert types
cat_features = ["Sex", "Embarked", "Pclass", "Title", "WomanOrChild"]
for col in cat_features:
    X[col] = X[col].astype("category")

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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

# Predict probabilities on hold-out validation set
cb_val_probs = cb_model.predict_proba(X_val)[:, 1]
xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]

# Ensemble predictions (simple average)
ensemble_val_probs = (cb_val_probs + xgb_val_probs) / 2.0
val_preds = (ensemble_val_probs >= 0.5).astype(int)

final_validation_score = accuracy_score(y_val, val_preds)

print(f"Final Validation Performance: {final_validation_score}")