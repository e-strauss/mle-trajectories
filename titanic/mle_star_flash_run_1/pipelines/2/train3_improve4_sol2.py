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

# Extract Deck from Cabin (first letter, handling missing as 'Unknown')
if "Cabin" in df.columns:
    deck = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) and str(x) != "" else "Unknown")
else:
    deck = pd.Series("Unknown", index=df.index)

# Extract and refine Title from Name
if "Name" in df.columns:
    title = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    common_titles = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master"}
    title = title.map(lambda t: common_titles.get(t, "Other")).fillna("Other")
else:
    title = pd.Series("Other", index=df.index)

# Conditional Age imputation based on Title and Pclass medians
age_imputed = df["Age"].copy()
age_medians = df.groupby([title, df["Pclass"]])["Age"].transform("median")
age_imputed = age_imputed.fillna(age_medians).fillna(df["Age"].median())

# Model non-linear family survival dynamics
family_size = df["SibSp"] + df["Parch"] + 1
family_tier = pd.Series("Alone", index=df.index)
family_tier[(family_size >= 2) & (family_size <= 4)] = "SmallFamily"
family_tier[family_size >= 5] = "LargeFamily"

# Compound categorical feature Sex_Pclass
sex_pclass = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)

# Log-transformed Fare
fare_imputed = df["Fare"].fillna(df["Fare"].median()).clip(lower=0)
log_fare = np.log1p(fare_imputed)

# Ticket sharing counts
if "Ticket" in df.columns:
    ticket_count = df["Ticket"].map(df["Ticket"].value_counts()).fillna(1)
else:
    ticket_count = pd.Series(1, index=df.index)

# Assemble feature DataFrame
X = pd.DataFrame({
    "Pclass": df["Pclass"],
    "Sex": df["Sex"],
    "Age": age_imputed,
    "SibSp": df["SibSp"],
    "Parch": df["Parch"],
    "LogFare": log_fare,
    "Embarked": df["Embarked"].fillna("Missing"),
    "Deck": deck,
    "Title": title,
    "FamilySize": family_size,
    "FamilyTier": family_tier,
    "Sex_Pclass": sex_pclass,
    "Ticket_Count": ticket_count
}, index=df.index)

target = "Survived"
y = df[target].copy()

# Set categorical types
cat_features = ["Pclass", "Sex", "Embarked", "Deck", "Title", "FamilyTier", "Sex_Pclass"]
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