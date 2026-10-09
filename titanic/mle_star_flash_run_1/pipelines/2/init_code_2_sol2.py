import pandas as pd
from lightgbm import LGBMClassifier
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

# Set categorical columns as pandas category dtype
cat_features = ["Sex", "Embarked", "Pclass"]
for col in cat_features:
    X[col] = X[col].astype("category")

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize LightGBMClassifier
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Train the model
model.fit(X_train, y_train)

# Evaluate on hold-out validation set
val_preds = model.predict(X_val)
final_validation_score = accuracy_score(y_val, val_preds)

print(f"Final Validation Performance: {final_validation_score}")