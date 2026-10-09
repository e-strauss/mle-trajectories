import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load dataset
train_path = "./input/train.csv"
df = pd.read_csv(train_path)
target = "Survived"


def evaluate_pipeline(feature_cols, cat_cols):
    X = df[feature_cols].copy()
    y = df[target].copy()

    for col in cat_cols:
        if col in X.columns:
            if col == "Embarked":
                X[col] = X[col].fillna("Missing")
            X[col] = X[col].astype("category")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cb_cats = [c for c in cat_cols if c in feature_cols]
    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=cb_cats if len(cb_cats) > 0 else None,
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
    val_preds = (ensemble_val_probs >= 0.5).astype(int)

    return accuracy_score(y_val, val_preds)


# 1. Baseline Full Pipeline
base_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
base_cats = ["Sex", "Embarked", "Pclass"]
baseline_score = evaluate_pipeline(base_features, base_cats)

# 2. Ablation 1: Drop Family Features ('SibSp', 'Parch')
ablation_1_features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
ablation_1_cats = ["Sex", "Embarked", "Pclass"]
score_ablation_1 = evaluate_pipeline(ablation_1_features, ablation_1_cats)

# 3. Ablation 2: Drop Embarkation Port ('Embarked')
ablation_2_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
ablation_2_cats = ["Sex", "Pclass"]
score_ablation_2 = evaluate_pipeline(ablation_2_features, ablation_2_cats)

# 4. Ablation 3: Drop Continuous Numerical Signals ('Age', 'Fare')
ablation_3_features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]
ablation_3_cats = ["Sex", "Embarked", "Pclass"]
score_ablation_3 = evaluate_pipeline(ablation_3_features, ablation_3_cats)

# Print Summary Table
results = {
    "Full Pipeline (Baseline)": baseline_score,
    "Ablation 1 (Drop Family Features SibSp & Parch)": score_ablation_1,
    "Ablation 2 (Drop Embarkation Port)": score_ablation_2,
    "Ablation 3 (Drop Continuous Signals Age & Fare)": score_ablation_3,
}

print("=" * 70)
print("ABLATION STUDY RESULTS")
print("=" * 70)
print(f"{'Experiment':<50} | {'Accuracy':<10} | {'Delta vs Base':<12}")
print("-" * 70)

drops = {}
for name, score in results.items():
    delta = score - baseline_score
    print(f"{name:<50} | {score:<10.5f} | {delta:+.5f}")
    if name != "Full Pipeline (Baseline)":
        drops[name] = delta

print("=" * 70)

# Determine the most critical component
most_impactful_ablation = min(drops, key=drops.get)
max_loss = abs(drops[most_impactful_ablation])

print(
    f"\nConclusion: The component whose removal causes the largest performance drop is:"
)
print(
    f"'{most_impactful_ablation}' with a drop of {max_loss:.5f} accuracy points."
)
print(
    "This indicates that this feature component contributes the most to overall performance."
)