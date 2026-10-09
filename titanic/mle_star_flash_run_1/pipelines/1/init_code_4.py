import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

train_df = pd.read_csv("./input/train.csv")


def prepare_data(df):
    df = df.copy()
    df["Title"] = (
        df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Unknown")
    )
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Embarked"] = df["Embarked"].fillna("Missing")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    cat_cols = ["Pclass", "Sex", "Embarked", "Title"]
    cat_dims = []
    cat_arrays = []
    for col in cat_cols:
        le = LabelEncoder()
        cat_arrays.append(le.fit_transform(df[col].astype(str)))
        cat_dims.append(len(le.classes_))

    x_cat = np.stack(cat_arrays, axis=1).astype(np.int64)

    num_cols = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
    scaler = StandardScaler()
    x_num = scaler.fit_transform(df[num_cols].values.astype(np.float32))

    return x_cat, x_num, cat_dims


X_cat, X_num, cat_dims = prepare_data(train_df)
y = train_df["Survived"].values

indices = np.arange(len(y))
idx_train, idx_val, y_train, y_val = train_test_split(
    indices, y, test_size=0.2, random_state=42, stratify=y
)

X_cat_train, X_cat_val = X_cat[idx_train], X_cat[idx_val]
X_num_train, X_num_val = X_num[idx_train], X_num[idx_val]

emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=[64, 32]):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_c, e_dim) for num_c, e_dim in zip(cat_dims, emb_dims)]
        )
        total_emb_dim = sum(emb_dims)
        in_dim = total_emb_dim + num_features

        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.network(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TabularNeuralNet(
    cat_dims, emb_dims, num_features=X_num.shape[1]
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=10
)

# Convert to torch tensors
t_X_cat_train = torch.tensor(X_cat_train, dtype=torch.long, device=device)
t_X_num_train = torch.tensor(X_num_train, dtype=torch.float32, device=device)
t_y_train = torch.tensor(y_train, dtype=torch.long, device=device)

t_X_cat_val = torch.tensor(X_cat_val, dtype=torch.long, device=device)
t_X_num_val = torch.tensor(X_num_val, dtype=torch.float32, device=device)
t_y_val = torch.tensor(y_val, dtype=torch.long, device=device)

best_val_acc = 0.0
batch_size = 32
num_epochs = 150

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(t_X_cat_train.size(0))

    for i in range(0, t_X_cat_train.size(0), batch_size):
        indices_batch = permutation[i : i + batch_size]
        batch_cat = t_X_cat_train[indices_batch]
        batch_num = t_X_num_train[indices_batch]
        batch_y = t_y_train[indices_batch]

        optimizer.zero_grad()
        outputs = model(batch_cat, batch_num)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(t_X_cat_val, t_X_num_val)
        val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
        val_acc = accuracy_score(y_val, val_preds)

    scheduler.step(val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc

model.eval()
with torch.no_grad():
    val_outputs = model(t_X_cat_val, t_X_num_val)
    preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

final_validation_score = accuracy_score(y_val, preds)
print(f"Final Validation Performance: {final_validation_score}")