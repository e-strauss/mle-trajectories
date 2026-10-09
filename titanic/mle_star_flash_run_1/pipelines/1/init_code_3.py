import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def prepare_data(df):
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Age"] = df["Age"].fillna(df.groupby("Title")["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    for col in ["Sex", "Embarked", "Title"]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

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
    ]
    return df[features]


train_df = pd.read_csv("./input/train.csv")

X = prepare_data(train_df)
y = train_df["Survived"]

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=350,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42,
)

model.fit(X_tr, y_tr)

preds = model.predict(X_val)
final_validation_score = accuracy_score(y_val, preds)

print(f"Final Validation Performance: {final_validation_score}")