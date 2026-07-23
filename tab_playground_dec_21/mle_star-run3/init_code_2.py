
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the training data
try:
    train_df = pd.read_csv("./input/train.csv")
except FileNotFoundError:
    print("Error: train.csv not found in the ./input directory. Please ensure the data is correctly placed.")
    train_df = None # Setting to None to prevent further execution if the file is missing

# Only proceed with the machine learning pipeline if the training data was loaded successfully
if train_df is not None:
    # Separate features and target
    X = train_df.drop(["Id", "Cover_Type"], axis=1)
    y = train_df["Cover_Type"]

    # Feature Engineering
    X['Hydrology_Distance'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 + X['Vertical_Distance_To_Hydrology']**2)
    X['Hillshade_Index'] = X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']
    X['Elevation_x_Slope'] = X['Elevation'] * X['Slope']

    # Define numerical features for scaling.
    # Wilderness_Area and Soil_Type columns are already binary (one-hot encoded) and generally
    # do not require scaling when used with tree-based models, and it's best to keep them as is.
    numerical_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                          'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                          'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                          'Horizontal_Distance_To_Fire_Points',
                          'Hydrology_Distance', 'Hillshade_Index', 'Elevation_x_Slope']

    # Apply Standard Scaling to numerical features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Model Training and Evaluation using 3-Fold Cross-Validation
    # Using RandomForestClassifier with default hyperparameters and a fixed random_state for reproducibility.
    # n_jobs=-1 is used to utilize all available CPU cores for faster computation.
    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Set up 3-Fold Stratified Cross-Validation
    # StratifiedKFold ensures that each fold has approximately the same percentage of samples of each target class.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("Starting 3-Fold Cross-Validation...")
    # Perform cross-validation to evaluate the model's performance
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"Cross-validation scores: {scores}")
    final_validation_score = np.mean(scores)
    print(f"Final Validation Performance: {final_validation_score}")
