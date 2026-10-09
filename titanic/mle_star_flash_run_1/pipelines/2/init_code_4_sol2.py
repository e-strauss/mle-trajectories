import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Load training data
train_path = "./input/train.csv"
df = pd.read_csv(train_path)

# Define feature columns and target
num_cols = ["Age", "Fare", "SibSp", "Parch"]
cat_cols = ["Sex", "Embarked", "Pclass"]
target_col = "Survived"

features = num_cols + cat_cols
X = df[features].copy()
y = df[target_col].copy()

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            ),
            cat_cols,
        ),
    ]
)

# Define classifier
clf = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)

# Create pipeline
pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", clf),
    ]
)

# Fit model on training data
pipe.fit(X_train, y_train)

# Evaluate on hold-out validation set
val_preds = pipe.predict(X_val)
final_validation_score = accuracy_score(y_val, val_preds)

print(f"Final Validation Performance: {final_validation_score}")