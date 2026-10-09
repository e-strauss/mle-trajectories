import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def prepare_data(df):
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    cat_cols = ["Sex", "Embarked", "Title", "Pclass"]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone",
    ]
    return df[features]


train_df = pd.read_csv("./input/train.csv")

X = prepare_data(train_df)
y = train_df["Survived"]

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

model.fit(X_tr, y_tr)

preds = model.predict(X_val)
final_validation_score = accuracy_score(y_val, preds)

print(f"Final Validation Performance: {final_validation_score}")