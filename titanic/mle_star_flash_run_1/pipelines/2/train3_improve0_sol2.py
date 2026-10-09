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

# Work on a copy of df to engineer domain features
df_feat = df.copy()

# Extract Title from Name
if "Name" in df_feat.columns:
    df_feat["Title"] = df_feat["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"]
    df_feat["Title"] = df_feat["Title"].replace(rare_titles, "Rare")
    df_feat["Title"] = df_feat["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df_feat["Title"] = df_feat["Title"].fillna("Missing")
else:
    df_feat["Title"] = "Missing"

# Family features
df_feat["FamilySize"] = df_feat["SibSp"] + df_feat["Parch"] + 1
df_feat["IsAlone"] = (df_feat["FamilySize"] == 1).astype(int)

# Fare handling & FarePerPerson
if "Fare" in df_feat.columns:
    df_feat["Fare"] = df_feat["Fare"].fillna(df_feat.groupby("Pclass")["Fare"].transform("median"))
    df_feat["FarePerPerson"] = df_feat["Fare"] / df_feat["FamilySize"]
else:
    df_feat["Fare"] = 0.0
    df_feat["FarePerPerson"] = 0.0

# Extract Deck from Cabin
if "Cabin" in df_feat.columns:
    df_feat["Deck"] = df_feat["Cabin"].astype(str).str[0]
    valid_decks = ["A", "B", "C", "D", "E", "F", "G", "T"]
    df_feat["Deck"] = df_feat["Deck"].apply(lambda x: x if x in valid_decks else "Missing")
else:
    df_feat["Deck"] = "Missing"

# Impute missing Age values conditional on Title and Pclass
age_group_medians = df_feat.groupby(["Title", "Pclass"])["Age"].transform("median")
df_feat["Age"] = df_feat["Age"].fillna(age_group_medians)
df_feat["Age"] = df_feat["Age"].fillna(df_feat.groupby("Pclass")["Age"].transform("median"))
df_feat["Age"] = df_feat["Age"].fillna(df_feat["Age"].median())

# Handle Embarked missing values
df_feat["Embarked"] = df_feat["Embarked"].fillna("Missing")

# Feature selection
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FarePerPerson",
    "Embarked",
    "Title",
    "FamilySize",
    "IsAlone",
    "Deck",
]
target = "Survived"

X = df_feat[features].copy()
y = df_feat[target].copy()

# Register categorical features for gradient boosting models
cat_features = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
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