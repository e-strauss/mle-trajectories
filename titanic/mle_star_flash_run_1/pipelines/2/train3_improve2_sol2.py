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

X = pd.DataFrame(index=df.index)

# 1. Base numerical features and family / ticket group size
family_size = df["SibSp"] + df["Parch"] + 1
ticket_counts = df["Ticket"].map(df["Ticket"].value_counts())
group_size = np.maximum(family_size, ticket_counts)

X["Age"] = df["Age"]
X["SibSp"] = df["SibSp"]
X["Parch"] = df["Parch"]
X["Fare"] = df["Fare"]
X["Fare_Per_Person"] = df["Fare"] / group_size
X["GroupSize"] = group_size

# Discretize GroupSize into regime bins
group_size_bin = pd.cut(
    group_size,
    bins=[0, 1, 4, np.inf],
    labels=["Alone", "SmallGroup", "LargeGroup"]
).astype(str)
X["GroupSizeBin"] = group_size_bin

# 2. Cabin room count and deck tier extraction
cabin_str = df["Cabin"].fillna("")
X["CabinRoomCount"] = cabin_str.apply(lambda c: len(c.split()) if c else 0)

raw_deck = cabin_str.apply(lambda c: c[0] if c else "Unknown")
deck_tier_map = {
    "A": "Upper_ABC", "B": "Upper_ABC", "C": "Upper_ABC",
    "D": "Mid_DE", "E": "Mid_DE",
    "F": "Lower_FG", "G": "Lower_FG", "T": "Lower_FG",
    "Unknown": "Unknown"
}
X["DeckTier"] = raw_deck.map(lambda d: deck_tier_map.get(d, "Unknown"))

# 3. Ticket prefix extraction
def extract_ticket_prefix(ticket):
    if pd.isna(ticket):
        return "UNKNOWN"
    ticket_clean = str(ticket).replace(".", "").replace("/", "").strip().split()
    return ticket_clean[0].upper() if len(ticket_clean) > 1 else "NUMERIC"

X["TicketPrefix"] = df["Ticket"].apply(extract_ticket_prefix)

# 4. Title extraction and consolidation
raw_title = df["Name"].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip().fillna("Unknown")
title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Dr": "Officer", "Rev": "Officer", "Col": "Officer", "Major": "Officer", "Capt": "Officer",
    "Don": "Royalty", "Dona": "Royalty", "Sir": "Royalty", "Lady": "Royalty",
    "Countess": "Royalty", "Jonkheer": "Royalty"
}
title_consolidated = raw_title.map(lambda t: title_map.get(t, "Other"))
X["Title"] = title_consolidated

# 5. Base categoricals and interaction features
X["Pclass"] = df["Pclass"].astype(str)
X["Sex"] = df["Sex"].astype(str)
X["Embarked"] = df["Embarked"].fillna("Missing").astype(str)

X["Title_Pclass"] = X["Title"] + "_" + X["Pclass"]
X["Sex_GroupSizeBin"] = X["Sex"] + "_" + X["GroupSizeBin"]

# Format all categorical columns for gradient boosting models
cat_features = [
    "Pclass", "Sex", "Embarked", "GroupSizeBin",
    "DeckTier", "TicketPrefix", "Title",
    "Title_Pclass", "Sex_GroupSizeBin"
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