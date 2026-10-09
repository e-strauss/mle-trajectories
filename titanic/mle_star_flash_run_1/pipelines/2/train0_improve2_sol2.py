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

# Extract spatial Cabin Deck feature
deck = df["Cabin"].fillna("Unknown").astype(str).str[0]
deck = deck.replace({"n": "Unknown"})

# Standardized Ticket prefix categories
clean_ticket = df["Ticket"].astype(str).str.replace(r"[./]", "", regex=True).str.strip().str.upper()
ticket_prefix = clean_ticket.str.extract(r"^([A-Z]+)", expand=False).fillna("NONE")
ticket_num = clean_ticket.str.extract(r"(\d+)$", expand=False).fillna("0")

# Unified Travel Group identifier
surname = df["Name"].astype(str).str.extract(r"^([^,]+)", expand=False).fillna("Unknown")
group_id = surname + "_" + ticket_num

# Group-level aggregations and party statistics
group_size = group_id.map(group_id.value_counts())
ticket_size = df["Ticket"].map(df["Ticket"].value_counts())
fare_imputed = df["Fare"].fillna(df["Fare"].median())
adjusted_fare = fare_imputed / group_size

# Construct feature set
X = pd.DataFrame(
    {
        "Pclass": df["Pclass"],
        "Sex": df["Sex"],
        "Age": df["Age"],
        "SibSp": df["SibSp"],
        "Parch": df["Parch"],
        "Fare": fare_imputed,
        "Embarked": df["Embarked"].fillna("Missing"),
        "Deck": deck,
        "TicketPrefix": ticket_prefix,
        "GroupID": group_id,
        "GroupSize": group_size,
        "TicketGroupSize": ticket_size,
        "IsMultiPerson": (group_size > 1).astype(int),
        "AdjustedFare": adjusted_fare,
        "Log_AdjustedFare": np.log1p(np.maximum(0, adjusted_fare)),
        "Age_Pclass": df["Age"] * df["Pclass"],
    },
    index=df.index,
)

# Register categorical features
cat_features = ["Sex", "Embarked", "Pclass", "Deck", "TicketPrefix", "GroupID"]
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