
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Set a seed for reproducibility across all libraries
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def get_device():
    """Gets the appropriate device (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feature_engineer(df, train_stats=None):
    """
    Engineers features for the model.

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
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    cscl_path = './input/nyc_cscl.csv'
    if os.path.exists(cscl_path):
        cscl = pd.read_csv(cscl_path, on_bad_lines='skip', low_memory=False)
        cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
        cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()

        df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
        df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
        df_engineered['boroname'].fillna('Unknown', inplace=True)
        df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
    else:
        df_engineered['boroname'] = 'Unknown'

    # --- Create Aggregate Features ---
    if train_stats is None:
        # Calculate stats from the dataframe itself (training mode)
        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']
        
        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std'])
        violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']
        
        boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std'])
        boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std']
        
        stats = {
            'street_agg': street_agg,
            'violation_agg': violation_agg,
            'boro_agg': boro_agg
        }
    else:
        # Apply pre-calculated stats (validation/test mode)
        stats = train_stats
        street_agg = stats['street_agg']
        violation_agg = stats['violation_agg']
        boro_agg = stats['boro_agg']

    # Merge aggregates
    df_engineered = pd.merge(df_engineered, street_agg, on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, violation_agg, on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, boro_agg, on='boroname', how='left')

    # Fill NaNs created by left merges (for unseen keys) and from std calc (where count=1)
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats


class LabelEncoderHandler:
    """A class to handle label encoding and unseen values."""
    def __init__(self, cols):
        self.cols = cols
        self.encoders = {col: LabelEncoder() for col in cols}

    def fit(self, df):
        for col in self.cols:
            # Add a placeholder for unseen values
            s = df[col].astype(str).copy()
            self.encoders[col].fit(s)
    
    def transform(self, df):
        df_transformed = df.copy()
        for col in self.cols:
            le = self.encoders[col]
            # Handle unseen values by assigning them to a new index
            known_classes = set(le.classes_)
            df_transformed[col] = df_transformed[col].astype(str).apply(lambda s: s if s in known_classes else '<unknown>')
            if '<unknown>' not in le.classes_:
                le.classes_ = np.append(le.classes_, '<unknown>')
            df_transformed[col] = le.transform(df_transformed[col])
        return df_transformed

    def get_embedding_sizes(self):
        """Get embedding sizes for the model, accounting for unseen values."""
        embedding_sizes = []
        for col in self.cols:
            # Number of categories +1 for the <unknown> class
            num_cats = len(self.encoders[col].classes_)
            emb_dim = min(50, (num_cats + 1) // 2)
            embedding_sizes.append((num_cats, emb_dim))
        return embedding_sizes


class TabularNN(nn.Module):
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
        # Apply ReLU to ensure non-negative output
        x = self.act(self.lin3(x))
        return x


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.01):
    """Trains the PyTorch model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.to(device)
    
    best_val_rmse = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_cat, x_cont, y_batch in train_loader:
            x_cat, x_cont, y_batch = x_cat.to(device), x_cont.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_cat, x_cont)
            loss = torch.sqrt(criterion(y_pred, y_batch))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
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
            torch.save(model.state_dict(), 'best_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val RMSE: {val_rmse:.4f}')

    # Load the best model found during training
    model.load_state_dict(torch.load('best_model.pth'))
    return model


def main():
    parser = argparse.ArgumentParser(description="Predict NYC parking violations using a Neural Network.")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv', help='Path to training data.')
    parser.add_argument('--test-path', type=str, default=None, help='(Optional) Path to test data.')
    args, _ = parser.parse_known_args()

    # --- 1. Load and Split Data ---
    print(f"Loading training data from {args.train_path}...")
    try:
        df_original = pd.read_csv(args.train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_path}. Exiting.")
        return

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples.")

    # --- 2. Feature Engineering ---
    print("Engineering features...")
    train_featured, train_stats = feature_engineer(train_df)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)
    
    # --- 3. Categorical & Continuous Feature Handling ---
    cat_features = ['street_name', 'violation_description', 'boroname']
    cont_features = [col for col in train_featured.columns if col not in cat_features + ['violation_count']]
    
    encoder_handler = LabelEncoderHandler(cat_features)
    encoder_handler.fit(train_featured)
    
    train_encoded = encoder_handler.transform(train_featured)
    val_encoded = encoder_handler.transform(val_featured)
    
    # --- 4. Prepare Tensors and DataLoaders ---
    X_cat_train = torch.tensor(train_encoded[cat_features].values, dtype=torch.long)
    X_cont_train = torch.tensor(train_encoded[cont_features].values, dtype=torch.float32)
    y_train = torch.tensor(train_encoded['violation_count'].values, dtype=torch.float32).view(-1, 1)

    X_cat_val = torch.tensor(val_encoded[cat_features].values, dtype=torch.long)
    X_cont_val = torch.tensor(val_encoded[cont_features].values, dtype=torch.float32)
    y_val = torch.tensor(val_encoded['violation_count'].values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_cat_train, X_cont_train, y_train)
    val_dataset = TensorDataset(X_cat_val, X_cont_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096)

    # --- 5. Model Initialization and Training ---
    print("Initializing and training the model...")
    device = get_device()
    print(f"Using device: {device}")
    
    embedding_sizes = encoder_handler.get_embedding_sizes()
    model = TabularNN(embedding_sizes, len(cont_features))
    
    model = train_model(model, train_loader, val_loader, device, epochs=50, lr=0.01)

    # --- 6. Final Validation ---
    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for x_cat_v, x_cont_v, _ in val_loader:
            x_cat_v, x_cont_v = x_cat_v.to(device), x_cont_v.to(device)
            preds = model(x_cat_v, x_cont_v)
            val_preds_list.append(preds.cpu().numpy())
    
    val_preds = np.concatenate(val_preds_list)
    val_preds[val_preds < 0] = 0 # Ensure non-negativity
    final_validation_score = np.sqrt(mean_squared_error(y_val.numpy(), val_preds))
    print(f'Final Validation Performance: {final_validation_score}')

    # --- 7. Test Prediction ---
    if args.test_path:
        print(f"\nProcessing test file: {args.test_path}")
        try:
            test_df_original = pd.read_csv(args.test_path)
        except FileNotFoundError:
            print(f"Error: Test file not found at {args.test_path}.")
            return
        
        submission_keys = test_df_original[['Street Name', 'Violation Description']].copy()
        test_ground_truth = test_df_original.get('violation_count')

        # Feature Engineering and Encoding
        test_featured, _ = feature_engineer(test_df_original, train_stats=train_stats)
        test_encoded = encoder_handler.transform(test_featured)

        X_cat_test = torch.tensor(test_encoded[cat_features].values, dtype=torch.long)
        X_cont_test = torch.tensor(test_encoded[cont_features].values, dtype=torch.float32)
        
        test_dataset = TensorDataset(X_cat_test, X_cont_test)
        test_loader = DataLoader(test_dataset, batch_size=4096)
        
        # Report on unseen keys
        train_keys = set(zip(df_original['Street Name'], df_original['Violation Description']))
        test_keys = set(zip(test_df_original['Street Name'], test_df_original['Violation Description']))
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Found {unseen_keys_count} (street_name, violation_type) pairs in test set not present in training data.")
        print("These pairs were handled by assigning them to a special '<unknown>' category and using aggregate features from broader groups.")

        # Prediction
        print("Generating predictions on the test set...")
        model.eval()
        test_preds_list = []
        with torch.no_grad():
            for x_cat_t, x_cont_t in test_loader:
                x_cat_t, x_cont_t = x_cat_t.to(device), x_cont_t.to(device)
                preds = model(x_cat_t, x_cont_t)
                test_preds_list.append(preds.cpu().numpy())
        
        test_preds = np.concatenate(test_preds_list)
        test_preds[test_preds < 0] = 0
        test_preds = np.round(test_preds).astype(int)

        # Create submission file
        submission_df = submission_keys
        submission_df.columns = ['street_name', 'violation_type']
        submission_df['predicted_count'] = test_preds.flatten()
        submission_df.to_csv('submission.csv', index=False)
        print("Successfully created submission.csv")

        if test_ground_truth is not None:
            test_rmse = np.sqrt(mean_squared_error(test_ground_truth, test_preds))
            print(f"RMSE on provided test set: {test_rmse:.4f}")

if __name__ == '__main__':
    main()
