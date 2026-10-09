import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load dataset
train_path = "./input/train.csv"
df = pd.read_csv(train_path)

import numpy as np
import pandas as pd

target = "Survived"
y = df[target].copy() if target in df.columns else None

# Make a copy of df to build feature matrix
X = pd.DataFrame(index=df.index)

# 1. Title extraction & categorization
titles = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_mapping = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Officer",
    "Rev": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Capt": "Officer",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Lady": "Royalty",
    "Countess": "Royalty",
    "Jonkheer": "Royalty",
    "Dona": "Royalty",
}
X["Title"] = titles.map(title_mapping).fillna("Other")

# 2. Conditional median age imputation by Title and Pclass
age_series = df["Age"].copy()
median_ages = df.assign(Title=X["Title"]).groupby(["Title", "Pclass"])["Age"].transform("median")
overall_median_age = df["Age"].median()
X["Age"] = age_series.fillna(median_ages).fillna(overall_median_age)

# 3. Base numerical and demographic features
X["Pclass"] = df["Pclass"].copy()
X["Sex"] = df["Sex"].copy()
X["SibSp"] = df["SibSp"].copy()
X["Parch"] = df["Parch"].copy()

# 4. Ticket-level group features
ticket_counts = df["Ticket"].map(df["Ticket"].value_counts())
X["TicketGroupSize"] = ticket_counts.fillna(1).astype(int)

# 5. Clean Fare imputation, log-transform and FarePerTicketGroup
fare_series = df["Fare"].copy()
median_fare = df.groupby("Pclass")["Fare"].transform("median")
fare_clean = fare_series.fillna(median_fare).fillna(df["Fare"].median())

X["Fare"] = np.log1p(np.maximum(0, fare_clean))
X["FarePerTicketGroup"] = np.log1p(np.maximum(0, fare_clean / X["TicketGroupSize"]))

# 6. Binary Cabin indicator (avoiding sparse deck splits)
X["HasCabin"] = df["Cabin"].notna().astype(int)

# 7. Embarked with missing value handling
X["Embarked"] = df["Embarked"].fillna("Missing")

# 8. High-signal domain interaction terms and flags
X["Sex_Pclass"] = X["Sex"].astype(str) + "_" + X["Pclass"].astype(str)
X["IsChild"] = (X["Age"] < 12).astype(int)
X["IsMother"] = ((X["Sex"] == "female") & (X["Parch"] > 0) & (X["Age"] > 18) & (X["Title"] == "Mrs")).astype(int)

# 9. Format categorical columns for tree models
cat_features = ["Sex", "Embarked", "Pclass", "Title", "Sex_Pclass"]
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