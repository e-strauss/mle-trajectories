import os
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
train_path = "./input/train.csv"
df = pd.read_csv(train_path)

# Feature selection
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

X = df[features].copy()
y = df[target].copy()

# Handle categorical missing values and types
cat_features = ["Sex", "Embarked", "Pclass"]
X["Embarked"] = X["Embarked"].fillna("Missing")
X[cat_features] = X[cat_features].astype(str)

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize CatBoostClassifier
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    eval_metric="Accuracy",
    random_seed=42,
    verbose=0,
)

# Train the model
model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

# Evaluate on hold-out validation set
val_preds = model.predict(X_val)
final_validation_score = accuracy_score(y_val, val_preds)

print(f"Final Validation Performance: {final_validation_score}")