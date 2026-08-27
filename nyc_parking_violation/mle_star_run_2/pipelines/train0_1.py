
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set a seed for reproducibility across all libraries
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def get_device():
    """Gets the appropriate device (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feature_engineer(df, train_stats=None):
    """
    Engineers features for the model.

    This function performs two main tasks:
    1. Augments the data with borough information by joining with an external file.
    2. Creates aggregate features (mean, sum, std, count) based on street,
       violation type, and borough.

    To prevent data leakage, it can operate in two modes:
    - Training mode (train_stats is None): Calculates and returns new statistics.
    - Inference mode (train_stats is provided): Applies pre-calculated statistics
      to a new dataset (validation or test).

    Args:
        df (pd.DataFrame): The dataframe to engineer features for.
        train_stats (dict, optional): A dictionary containing statistics (aggregates)
                                      from the training set. If None, stats are
                                      calculated from df itself.

    Returns:
        pd.DataFrame: The dataframe with new features.
        dict: A dictionary of the calculated stats (if train_stats was None).
    """
    df_engineered = df.copy()

    # Standardize column names for easier access
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    cscl_path = './input/nyc_cscl.csv'
    if os.path.exists(cscl_path):
        cscl = pd.read_csv(cscl_path, on_bad_lines='skip', low_memory=False)
        # Select relevant columns and create a unique mapping from street to borough
        cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
        cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()

        # Merge borough information
        df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
        df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
        df_engineered['boroname'].fillna('Unknown', inplace=True)
        df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
    else:
        # If the augmentation file is not present, create a placeholder column
        df_engineered['boroname'] = 'Unknown'

    # The target column might not be present in a keys-only test file
    has_target = 'violation_count' in df_engineered.columns

    # --- Create Aggregate Features ---
    if train_stats is None:
        # Training mode: Calculate stats from the dataframe itself
        if not has_target:
             raise ValueError("`violation_count` column is required to build training statistics.")
        
        # Aggregate by street name
        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']
        
        # Aggregate by violation description
        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std'])
        violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']
        
        # Aggregate by borough
        boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std'])
        boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std']
        
        # Store calculated stats for later use
        stats = {
            'street_agg': street_agg,
            'violation_agg': violation_agg,
            'boro_agg': boro_agg,
        }
    else:
        # Inference mode: Apply pre-calculated stats
        stats = train_stats

    # Merge aggregate features onto the dataframe
    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    # Fill NaNs created by left merges.
    # NaNs can occur for unseen keys (e.g., a new street in the test set)
    # or for std deviation where a group has only one member.
    # We fill with 0, assuming no prior information implies a zero effect.
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats

# --- Neural Network Components ---

class LabelEncoderHandler:
    """A class to handle label encoding and unseen values for the NN."""
    def __init__(self, cols):
        self.cols = cols
        self.encoders = {col: LabelEncoder() for col in cols}

    def fit(self, df):
        for col in self.cols:
            # Fit on the string representation of the column
            s = df[col].astype(str)
            self.encoders[col].fit(s)
    
    def transform(self, df):
        df_transformed = df.copy()
        for col in self.cols:
            le = self.encoders[col]
            known_classes = set(le.classes_)
            # Handle unseen values by mapping them to a special '<unknown>' token
            # Then transform to numerical labels
            transformed_series = df_transformed[col].astype(str).apply(lambda s: s if s in known_classes else '<unknown>')
            if '<unknown>' not in le.classes_:
                le.classes_ = np.append(le.classes_, '<unknown>')
            df_transformed[col] = le.transform(transformed_series)
        return df_transformed

    def get_embedding_sizes(self):
        """Get embedding sizes for the model, accounting for unseen values."""
        embedding_sizes = []
        for col in self.cols:
            num_cats = len(self.encoders[col].classes_)
            # Heuristic for embedding dimension
            emb_dim = min(50, (num_cats + 1) // 2)
            embedding_sizes.append((num_cats, emb_dim))
        return embedding_sizes


class TabularNN(nn.Module):
    """A simple feed-forward neural network for tabular data."""
    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(cats, size) for cats, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont = n_emb, n_cont
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, 1)
        self.bn_cont = nn.BatchNorm1d(self.n_cont) if self.n_cont > 0 else nn.Identity()
        self.dropout = nn.Dropout(0.3)
        self.act = nn.ReLU()

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        if self.n_cont > 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1)
        x = self.act(self.lin1(x))
        x = self.dropout(x)
        x = self.act(self.lin2(x))
        x = self.dropout(x)
        # Apply ReLU to ensure non-negative output, similar to clipping predictions
        x = self.act(self.lin3(x))
        return x


def train_nn_model(model, train_loader, val_loader, device, epochs=50, lr=0.01):
    """Trains the PyTorch model and returns the best model based on validation RMSE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.to(device)
    
    best_val_rmse = float('inf')

    for epoch in range(epochs):
        model.train()
        for x_cat, x_cont, y_batch in train_loader:
            x_cat, x_cont, y_batch = x_cat.to(device), x_cont.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_cat, x_cont)
            loss = torch.sqrt(criterion(y_pred, y_batch))
            loss.backward()
            optimizer.step()
        
        # Validation step
        model.eval()
        val_preds_list = []
        val_targets_list = []
        with torch.no_grad():
            for x_cat_val, x_cont_val, y_val_batch in val_loader:
                x_cat_val, x_cont_val = x_cat_val.to(device), x_cont_val.to(device)
                preds = model(x_cat_val, x_cont_val)
                val_preds_list.append(preds.cpu().numpy())
                val_targets_list.append(y_val_batch.cpu().numpy())

        val_preds = np.concatenate(val_preds_list)
        val_targets = np.concatenate(val_targets_list)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), 'best_nn_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f'NN Epoch {epoch+1}/{epochs}, Val RMSE: {val_rmse:.4f}')

    # Load the best model found during training
    model.load_state_dict(torch.load('best_nn_model.pth'))
    return model


def main():
    """
    Main function to run the training and prediction pipeline for an ensemble model.
    """
    parser = argparse.ArgumentParser(description="Predict NYC parking violations using an ensemble of Ridge Regression and a Neural Network.")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='(Optional) Path to the test/evaluation data CSV file.')
    args = parser.parse_args()

    # --- 1. Load Data ---
    print(f"Loading training data from {args.train_path}...")
    df_original = pd.read_csv(args.train_path)

    # --- 2. Validation Split ---
    print("Splitting data into train and validation sets...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples.")

    # --- 3. Feature Engineering (common for both models) ---
    print("Engineering features for training set...")
    train_featured, train_stats = feature_engineer(train_df)
    
    print("Engineering features for validation set...")
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    target = 'violation_count'
    y_train = train_featured[target]
    y_val = val_featured[target]

    # --- 4. Ridge Model Training & Validation ---
    print("\n--- Training Ridge Model ---")
    
    ridge_cat_features = ['violation_description', 'boroname']
    ridge_num_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    ridge_all_features = ridge_num_features + ridge_cat_features
    X_train_ridge = train_featured[ridge_all_features]
    X_val_ridge = val_featured[ridge_all_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ridge_num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ridge_cat_features)
        ],
        remainder='passthrough'
    )

    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])

    ridge_pipeline.fit(X_train_ridge, y_train)
    print(f"Ridge training complete. Best alpha: {ridge_pipeline.named_steps['regressor'].alpha_}")
    
    val_predictions_ridge = ridge_pipeline.predict(X_val_ridge)
    val_predictions_ridge[val_predictions_ridge < 0] = 0

    # --- 5. Neural Network Model Training & Validation ---
    print("\n--- Training Neural Network Model ---")
    
    nn_cat_features = ['street_name', 'violation_description', 'boroname']
    nn_cont_features = [col for col in train_featured.columns if col not in nn_cat_features + [target]]

    encoder_handler = LabelEncoderHandler(nn_cat_features)
    encoder_handler.fit(train_featured)
    
    train_encoded = encoder_handler.transform(train_featured)
    val_encoded = encoder_handler.transform(val_featured)
    
    X_cat_train = torch.tensor(train_encoded[nn_cat_features].values, dtype=torch.long)
    X_cont_train = torch.tensor(train_encoded[nn_cont_features].values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_cat_val = torch.tensor(val_encoded[nn_cat_features].values, dtype=torch.long)
    X_cont_val = torch.tensor(val_encoded[nn_cont_features].values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_cat_train, X_cont_train, y_train_tensor)
    val_dataset = TensorDataset(X_cat_val, X_cont_val, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096)
    
    device = get_device()
    print(f"Using device: {device}")
    
    embedding_sizes = encoder_handler.get_embedding_sizes()
    nn_model = TabularNN(embedding_sizes, len(nn_cont_features))
    
    nn_model = train_nn_model(nn_model, train_loader, val_loader, device, epochs=50, lr=0.01)

    nn_model.eval()
    val_preds_nn_list = []
    with torch.no_grad():
        for x_cat_v, x_cont_v, _ in val_loader:
            x_cat_v, x_cont_v = x_cat_v.to(device), x_cont_v.to(device)
            preds = nn_model(x_cat_v, x_cont_v)
            val_preds_nn_list.append(preds.cpu().numpy())
            
    val_predictions_nn = np.concatenate(val_preds_nn_list).flatten()
    val_predictions_nn[val_predictions_nn < 0] = 0

    # --- 6. Ensemble Validation ---
    print("\n--- Ensembling and Final Evaluation ---")
    
    # Simple averaging ensemble
    ensemble_val_predictions = (val_predictions_ridge + val_predictions_nn) / 2.0
    
    final_validation_score = np.sqrt(mean_squared_error(y_val, ensemble_val_predictions))
    print(f'Final Validation Performance: {final_validation_score:.4f}')

    # --- 7. Test Prediction (if applicable) ---
    if args.test_path:
        print(f"\nProcessing test file: {args.test_path}")
        test_df_original = pd.read_csv(args.test_path)
        
        submission_keys = test_df_original[['Street Name', 'Violation Description']].copy()
        test_ground_truth = test_df_original.get('violation_count')

        print("Engineering features for the test set...")
        test_featured, _ = feature_engineer(test_df_original, train_stats=train_stats)
        
        # Ridge predictions
        X_test_ridge = test_featured[ridge_all_features]
        test_predictions_ridge = ridge_pipeline.predict(X_test_ridge)
        test_predictions_ridge[test_predictions_ridge < 0] = 0

        # NN predictions
        test_encoded = encoder_handler.transform(test_featured)
        X_cat_test = torch.tensor(test_encoded[nn_cat_features].values, dtype=torch.long)
        X_cont_test = torch.tensor(test_encoded[nn_cont_features].values, dtype=torch.float32)
        test_dataset_nn = TensorDataset(X_cat_test, X_cont_test)
        test_loader_nn = DataLoader(test_dataset_nn, batch_size=4096)
        
        nn_model.eval()
        test_preds_nn_list = []
        with torch.no_grad():
            for x_cat_t, x_cont_t in test_loader_nn:
                x_cat_t, x_cont_t = x_cat_t.to(device), x_cont_t.to(device)
                preds = nn_model(x_cat_t, x_cont_t)
                test_preds_nn_list.append(preds.cpu().numpy())
                
        test_predictions_nn = np.concatenate(test_preds_nn_list).flatten()
        test_predictions_nn[test_predictions_nn < 0] = 0

        # Ensemble test predictions
        ensemble_test_predictions = (test_predictions_ridge + test_predictions_nn) / 2.0
        
        # Create submission file
        submission_df = submission_keys
        submission_df.columns = ['street_name', 'violation_type']
        submission_df['predicted_count'] = np.round(ensemble_test_predictions).astype(int)
        
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Successfully created {submission_path}")

        if test_ground_truth is not None:
            test_rmse = np.sqrt(mean_squared_error(test_ground_truth, ensemble_test_predictions))
            print(f"RMSE on provided test set: {test_rmse:.4f}")

if __name__ == '__main__':
    main()
