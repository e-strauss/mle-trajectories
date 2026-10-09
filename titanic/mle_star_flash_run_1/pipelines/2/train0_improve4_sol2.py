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
y = df[target].copy()

# Initialize feature dataframe
X = pd.DataFrame(index=df.index)

# Base numerical and demographic features
X["Pclass"] = df["Pclass"]
X["Sex"] = df["Sex"]
X["Age"] = df["Age"]
X["SibSp"] = df["SibSp"]
X["Parch"] = df["Parch"]
X["Fare"] = df["Fare"]
X["Embarked"] = df["Embarked"].fillna("Missing")

# Extract Title and Surname from Name
title = df["Name"].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
surname = df["Name"].str.split(",").str[0].str.strip()
X["Title"] = title.fillna("Missing")
X["Surname"] = surname.fillna("Missing")

# Extract standardized TicketPrefix and compute group frequencies
ticket_clean = df["Ticket"].astype(str).str.replace(r"[\./]", "", regex=True).str.strip()
ticket_prefix = ticket_clean.apply(lambda x: x.split()[0] if len(x.split()) > 1 else "None")
X["TicketPrefix"] = ticket_prefix

X["TicketFrequency"] = df["Ticket"].map(df["Ticket"].value_counts())
X["SurnameFrequency"] = surname.map(surname.value_counts())

# Relative economic indicators
X["LogFare"] = np.log1p(np.maximum(0, df["Fare"].fillna(df["Fare"].median())))
median_fare_by_pclass_embarked = df.groupby(["Pclass", "Embarked"])["Fare"].transform("median")
X["FareRatioByPclass"] = df["Fare"] / (median_fare_by_pclass_embarked + 1e-5)

# Demographic interactions
X["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
X["IsMother"] = ((df["Sex"] == "female") & (df["Parch"] > 0) & (df["Age"] > 18) & (title != "Miss")).astype(int)

# Encode categorical features
cat_features = ["Sex", "Embarked", "Pclass", "Sex_Pclass", "TicketPrefix", "Title", "Surname", "IsMother"]
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