import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
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

# Ensure working DataFrame format
if not isinstance(X, pd.DataFrame):
    X_df = pd.DataFrame(X)
else:
    X_df = X.copy()

y_arr = np.array(y).ravel()

# 1. Feature Engineering: High-signal Domain & Interaction Features
feat_df = pd.DataFrame(index=X_df.index)

# Shared Ticket frequency & Real Individual Fare
if "Ticket" in df.columns:
    ticket_counts = df["Ticket"].value_counts()
    feat_df["Ticket_Count"] = df["Ticket"].map(ticket_counts).fillna(1)
elif "Ticket" in X_df.columns:
    ticket_counts = X_df["Ticket"].value_counts()
    feat_df["Ticket_Count"] = X_df["Ticket"].map(ticket_counts).fillna(1)
elif "Ticket_Count" in X_df.columns:
    feat_df["Ticket_Count"] = X_df["Ticket_Count"]
else:
    feat_df["Ticket_Count"] = 1

if "Fare" in X_df.columns:
    fare_filled = pd.to_numeric(X_df["Fare"], errors="coerce").fillna(X_df["Fare"].median())
    feat_df["Real_Fare"] = fare_filled / feat_df["Ticket_Count"]
    feat_df["Log_Real_Fare"] = np.log1p(np.maximum(0, feat_df["Real_Fare"]))
elif "Real_Fare" in X_df.columns:
    feat_df["Real_Fare"] = pd.to_numeric(X_df["Real_Fare"], errors="coerce")
    feat_df["Log_Real_Fare"] = np.log1p(np.maximum(0, feat_df["Real_Fare"]))

# Pclass x Sex interaction (cast properly to numeric to avoid category addition errors)
if "Sex" in X_df.columns and "Pclass" in X_df.columns:
    sex_num = X_df["Sex"].astype(str).map({"male": 0, "female": 1}).fillna(0).astype(int)
    pclass_num = pd.to_numeric(X_df["Pclass"], errors="coerce").fillna(3).astype(int)
    feat_df["Sex_Num"] = sex_num
    feat_df["Pclass"] = pclass_num
    feat_df["Pclass_Sex"] = pclass_num * 2 + sex_num
elif "Sex_Num" in X_df.columns and "Pclass" in X_df.columns:
    feat_df["Sex_Num"] = pd.to_numeric(X_df["Sex_Num"], errors="coerce").fillna(0).astype(int)
    feat_df["Pclass"] = pd.to_numeric(X_df["Pclass"], errors="coerce").fillna(3).astype(int)
    feat_df["Pclass_Sex"] = feat_df["Pclass"] * 2 + feat_df["Sex_Num"]

# Family Size & Title extraction
if "SibSp" in X_df.columns and "Parch" in X_df.columns:
    sibsp_num = pd.to_numeric(X_df["SibSp"], errors="coerce").fillna(0)
    parch_num = pd.to_numeric(X_df["Parch"], errors="coerce").fillna(0)
    feat_df["Family_Size"] = sibsp_num + parch_num + 1
    feat_df["Is_Alone"] = (feat_df["Family_Size"] == 1).astype(int)

if "Age" in X_df.columns:
    age_num = pd.to_numeric(X_df["Age"], errors="coerce")
    feat_df["Age"] = age_num.fillna(age_num.median())
    feat_df["Is_Child"] = (feat_df["Age"] < 12).astype(int)

if "Name" in df.columns:
    feat_df["Surname"] = df["Name"].astype(str).apply(lambda x: x.split(",")[0].strip())
    titles = df["Name"].astype(str).str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3,
        "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 1, "Ms": 1, "Mme": 2
    }
    feat_df["Title_Code"] = titles.map(title_mapping).fillna(5).astype(int)
elif "Name" in X_df.columns:
    feat_df["Surname"] = X_df["Name"].astype(str).apply(lambda x: x.split(",")[0].strip())
    titles = X_df["Name"].astype(str).str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3,
        "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 1, "Ms": 1, "Mme": 2
    }
    feat_df["Title_Code"] = titles.map(title_mapping).fillna(5).astype(int)
else:
    feat_df["Surname"] = "Unknown"

# Fill remaining numeric columns if needed
for col in feat_df.select_dtypes(include=[np.number]).columns:
    feat_df[col] = feat_df[col].fillna(feat_df[col].median())

# Select top ~10 non-redundant compact features
candidate_features = [
    "Sex_Num", "Pclass", "Pclass_Sex", "Real_Fare", "Log_Real_Fare",
    "Family_Size", "Is_Alone", "Is_Child", "Age", "Title_Code"
]
selected_features = [f for f in candidate_features if f in feat_df.columns][:10]
X_clean = feat_df[selected_features].copy()

# 2. Repeated Stratified K-Fold CV (5 folds x 5 repeats = 25 evaluations)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

oof_preds_lgb = np.zeros(len(y_arr))
oof_preds_cat = np.zeros(len(y_arr))
oof_preds_lr = np.zeros(len(y_arr))
oof_counts = np.zeros(len(y_arr))

# 3. Model Trio & Within-fold Target Encoding for Family Survival Cues
for train_idx, val_idx in rskf.split(X_clean, y_arr):
    X_tr = X_clean.iloc[train_idx].copy()
    y_tr = y_arr[train_idx]
    X_va = X_clean.iloc[val_idx].copy()
    y_va = y_arr[val_idx]
    
    # Strictly in-fold surname / family survival rate encoding
    surnames_tr = feat_df["Surname"].iloc[train_idx]
    surnames_va = feat_df["Surname"].iloc[val_idx]
    
    global_mean = y_tr.mean()
    surname_stats = pd.DataFrame({"Surname": surnames_tr, "Target": y_tr}).groupby("Surname")["Target"].agg(["count", "mean"])
    # Credible family survival cue with smoothing
    smooth_weight = 3.0
    surname_stats["Survival_Cue"] = (surname_stats["count"] * surname_stats["mean"] + smooth_weight * global_mean) / (surname_stats["count"] + smooth_weight)
    cue_map = surname_stats["Survival_Cue"].to_dict()
    
    X_tr["Family_Survival_Cue"] = surnames_tr.map(cue_map).fillna(global_mean)
    X_va["Family_Survival_Cue"] = surnames_va.map(cue_map).fillna(global_mean)
    
    # 1. LightGBM (Regularized)
    clf_lgb = LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.04,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    clf_lgb.fit(X_tr, y_tr)
    oof_preds_lgb[val_idx] += clf_lgb.predict_proba(X_va)[:, 1]
    
    # 2. CatBoost (Regularized)
    clf_cat = CatBoostClassifier(
        iterations=150,
        depth=4,
        learning_rate=0.04,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=0
    )
    clf_cat.fit(X_tr, y_tr)
    oof_preds_cat[val_idx] += clf_cat.predict_proba(X_va)[:, 1]
    
    # 3. ElasticNet Logistic Regression
    clf_lr = Pipeline([
        ("scaler", RobustScaler()),
        ("logit", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.2,
            max_iter=1000,
            random_state=42
        ))
    ])
    clf_lr.fit(X_tr, y_tr)
    oof_preds_lr[val_idx] += clf_lr.predict_proba(X_va)[:, 1]
    
    oof_counts[val_idx] += 1

# Average across repeats
oof_preds_lgb /= oof_counts
oof_preds_cat /= oof_counts
oof_preds_lr /= oof_counts

# 4. Rank Averaging Ensemble
rank_lgb = rankdata(oof_preds_lgb) / len(oof_preds_lgb)
rank_cat = rankdata(oof_preds_cat) / len(oof_preds_cat)
rank_lr = rankdata(oof_preds_lr) / len(oof_preds_lr)

ensemble_rank_oof = 0.40 * rank_lgb + 0.40 * rank_cat + 0.20 * rank_lr

# Threshold Calibration for Accuracy Maximization
best_thresh = 0.5
best_acc = 0.0
for thresh in np.linspace(0.35, 0.65, 301):
    acc = accuracy_score(y_arr, (ensemble_rank_oof >= thresh).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

# Set final train/validation split artifacts for downstream pipeline compatibility
train_idx, val_idx = next(RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42).split(X_clean, y_arr))
X_train, X_val = X_clean.iloc[train_idx].copy(), X_clean.iloc[val_idx].copy()
y_train, y_val = y_arr[train_idx], y_arr[val_idx]

# Check for any valid categorical features in training set
valid_cat_features = [col for col in ["Pclass", "Sex_Num", "Title_Code"] if col in X_train.columns and str(X_train[col].dtype) == "category"]

# Initialize CatBoostClassifier
cb_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=valid_cat_features if len(valid_cat_features) > 0 else None,
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