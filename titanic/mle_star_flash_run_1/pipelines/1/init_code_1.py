import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def prepare_data(df):
    df = df.copy()
    df["Title"] = (
        df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Missing")
    )
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["CabinDeck"] = df["Cabin"].str[0].fillna("Missing")
    df["Embarked"] = df["Embarked"].fillna("Missing")
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
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
        "CabinDeck",
    ]
    return df[features]


train_df = pd.read_csv("./input/train.csv")

X = prepare_data(train_df)
y = train_df["Survived"]

cat_features = ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"]

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.03,
    depth=5,
    cat_features=cat_features,
    eval_metric="Accuracy",
    random_seed=42,
    verbose=0,
)

model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)

preds = model.predict(X_val)
final_validation_score = accuracy_score(y_val, preds)

print(f"Final Validation Performance: {final_validation_score}")