import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load dataset
train_path = "./input/train.csv"
df = pd.read_csv(train_path)

import re
import pandas as pd
import numpy as np

# Target
target = "Survived"
y = df[target].copy() if target in df.columns else None

# Missingness indicators
age_is_missing = df["Age"].isna().astype(int)
cabin_is_missing = df["Cabin"].isna().astype(int)

# Extract and group social titles
extracted_titles = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_mapping = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}
title = extracted_titles.map(title_mapping).fillna("Rare")

# Cabin attributes: Deck, cabin count, port/starboard parity
deck = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) else "Missing")
cabin_count = df["Cabin"].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)


def get_cabin_side(val):
    if pd.isna(val):
        return "Missing"
    numbers = re.findall(r"\d+", str(val))
    if numbers:
        return "Starboard" if int(numbers[0]) % 2 != 0 else "Port"
    return "Unknown"


cabin_side = df["Cabin"].apply(get_cabin_side)

# Family structure buckets
family_size = df["SibSp"] + df["Parch"] + 1


def get_family_type(size):
    if size == 1:
        return "Single"
    elif size <= 4:
        return "SmallFamily"
    return "LargeFamily"


family_type = family_size.apply(get_family_type)

# Demographic interaction and Ticket digit length
title_pclass = title.astype(str) + "_" + df["Pclass"].astype(str)
ticket_digit_len = (
    df["Ticket"].astype(str).apply(lambda x: len(re.sub(r"\D", "", x)))
)

# Build feature matrix
X = pd.DataFrame(
    {
        "Pclass": df["Pclass"],
        "Sex": df["Sex"],
        "Age": df["Age"],
        "SibSp": df["SibSp"],
        "Parch": df["Parch"],
        "Fare": df["Fare"],
        "Embarked": df["Embarked"].fillna("Missing"),
        "Age_is_missing": age_is_missing,
        "Cabin_is_missing": cabin_is_missing,
        "Title": title,
        "Deck": deck,
        "Cabin_count": cabin_count,
        "Cabin_side": cabin_side,
        "FamilyType": family_type,
        "Title_Pclass": title_pclass,
        "Ticket_digit_len": ticket_digit_len,
    }
)

# Register categorical columns for tree algorithms
cat_features = [
    "Title",
    "Title_Pclass",
    "Deck",
    "FamilyType",
    "Embarked",
    "Pclass",
    "Sex",
    "Cabin_side",
]
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