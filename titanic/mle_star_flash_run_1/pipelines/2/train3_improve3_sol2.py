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

# Extract and simplify Title from Name
title_raw = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_map = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
}
df_temp = df.copy()
df_temp["Title"] = title_raw.map(lambda x: title_map.get(x, "Rare" if pd.notna(x) else "Missing"))

# Conditional median imputation for missing Age based on Title and Pclass
age_imputed = df_temp.groupby(["Title", "Pclass"])["Age"].transform(lambda s: s.fillna(s.median()))
df_temp["Age"] = age_imputed.fillna(df_temp["Age"].median())

# Compute PerPersonFare using ticket frequency counts with log1p transformation
ticket_counts = df_temp["Ticket"].map(df_temp["Ticket"].value_counts()).fillna(1)
fare_filled = df_temp["Fare"].fillna(df_temp["Fare"].median())
df_temp["PerPersonFare_Log"] = np.log1p(fare_filled / ticket_counts)

# Targeted domain indicators and interaction terms
df_temp["Is_Woman_or_Child"] = (
    (df_temp["Sex"] == "female") | (df_temp["Age"] < 12) | (df_temp["Title"] == "Master")
).astype(int)
df_temp["Is_Alone"] = ((df_temp["SibSp"] + df_temp["Parch"]) == 0).astype(int)
df_temp["Age_Pclass"] = df_temp["Age"] * df_temp["Pclass"].astype(float)
df_temp["Embarked"] = df_temp["Embarked"].fillna("Missing")

# Select compact, high-signal feature set
features = [
    "Pclass",
    "Sex",
    "Title",
    "Age",
    "SibSp",
    "Parch",
    "PerPersonFare_Log",
    "Embarked",
    "Is_Woman_or_Child",
    "Is_Alone",
    "Age_Pclass",
]
target = "Survived"

X = df_temp[features].copy()
y = df_temp[target].copy()

# Set clean categorical types
cat_features = ["Pclass", "Sex", "Title", "Embarked"]
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