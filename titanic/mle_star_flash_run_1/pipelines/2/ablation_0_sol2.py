import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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
for col in cat_features:
    X[col] = X[col].astype("category")

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 1. Full Pipeline: CatBoost + XGBoost Ensemble
cb_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    eval_metric="Accuracy",
    random_seed=42,
    verbose=0,
)
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

cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
xgb_model.fit(X_train, y_train)

cb_val_probs = cb_model.predict_proba(X_val)[:, 1]
xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]

ensemble_val_probs = (cb_val_probs + xgb_val_probs) / 2.0
full_score = accuracy_score(y_val, (ensemble_val_probs >= 0.5).astype(int))

# 2. Ablation A: Remove XGBoost (CatBoost only)
cb_only_score = accuracy_score(y_val, (cb_val_probs >= 0.5).astype(int))

# 3. Ablation B: Remove CatBoost (XGBoost only)
xgb_only_score = accuracy_score(y_val, (xgb_val_probs >= 0.5).astype(int))

# 4. Ablation C: Remove Categorical Features (Evaluate on numerical features only)
num_features = ["Age", "SibSp", "Parch", "Fare"]
X_train_num = X_train[num_features].copy()
X_val_num = X_val[num_features].copy()

cb_num = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, eval_metric="Accuracy", random_seed=42, verbose=0)
xgb_num = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, eval_metric="logloss")

cb_num.fit(X_train_num, y_train, eval_set=(X_val_num, y_val), verbose=False)
xgb_num.fit(X_train_num, y_train)

num_cb_probs = cb_num.predict_proba(X_val_num)[:, 1]
num_xgb_probs = xgb_num.predict_proba(X_val_num)[:, 1]
num_ensemble_probs = (num_cb_probs + num_xgb_probs) / 2.0
no_cat_score = accuracy_score(y_val, (num_ensemble_probs >= 0.5).astype(int))

# Output Ablation Results
print("--- Ablation Study Results ---")
print(f"Full Ensemble (CatBoost + XGBoost + All Features): {full_score:.5f}")
print(f"Ablation 1 (CatBoost Only, drop XGBoost)          : {cb_only_score:.5f} (Δ = {cb_only_score - full_score:+.5f})")
print(f"Ablation 2 (XGBoost Only, drop CatBoost)          : {xgb_only_score:.5f} (Δ = {xgb_only_score - full_score:+.5f})")
print(f"Ablation 3 (Drop Categorical Features)            : {no_cat_score:.5f} (Δ = {no_cat_score - full_score:+.5f})")

# Determine the most critical component
drops = {
    "Categorical Features (Sex, Embarked, Pclass)": full_score - no_cat_score,
    "CatBoost Model": full_score - xgb_only_score,
    "XGBoost Model": full_score - cb_only_score,
}
most_critical = max(drops, key=drops.get)
print(f"\nConclusion: [{most_critical}] contributes the most to the overall performance with a performance drop of {drops[most_critical]:.5f} when removed.")