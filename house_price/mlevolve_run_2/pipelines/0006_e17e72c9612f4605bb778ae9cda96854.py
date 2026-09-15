import gc
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==============================================================================
# 1. PATHS AND DIRECTORY SETUP
# ==============================================================================
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


# ==============================================================================
# 2. EVALUATION METRICS
# ==============================================================================
def compute_log10_rmse(y_true, y_pred):
    """
    Computes official competition metric: RMSE on log10(price).
    Compatible with numpy arrays and torch tensors.
    """
    if isinstance(y_true, torch.Tensor):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def compute_raw_price_rmse(raw_true, raw_pred):
    """
    Computes official competition metric on raw currency prices with >= 1 clipping.
    """
    p_true = np.clip(np.asarray(raw_true, dtype=np.float64), 1.0, None)
    p_pred = np.clip(np.asarray(raw_pred, dtype=np.float64), 1.0, None)
    return float(np.sqrt(np.mean((np.log10(p_pred) - np.log10(p_true)) ** 2)))


# ==============================================================================
# 3. MODEL ARCHITECTURES
# ==============================================================================
def get_lgb_model_params():
    """
    Returns optimal LightGBM model hyperparameter configuration for log10 price regression.
    """
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 127,
        "max_depth": 12,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "n_estimators": 2500,
        "random_state": 42,
        "n_jobs": -1,
        "importance_type": "gain",
        "verbose": -1,
    }


def create_lgb_model():
    return lgb.LGBMRegressor(**get_lgb_model_params())


class TabularResidualBlock(nn.Module):
    """
    Residual MLP block with LayerNorm, SiLU activation, and Dropout regularization.
    """

    def __init__(self, hidden_dim, dropout_rate=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class PropertyPriceTabularNet(nn.Module):
    """
    Deep Tabular Neural Network with Entity Embeddings for high-cardinality
    geographic/property categoricals, continuous feature normalization, and
    residual representation layers for property transaction price prediction.
    """

    def __init__(
        self,
        num_continuous_features,
        cat_cardinalities,
        embedding_dims=None,
        hidden_dims=(256, 256, 128),
        dropout_rate=0.2,
    ):
        super().__init__()
        self.num_continuous = num_continuous_features
        self.embeddings = nn.ModuleList()
        total_embed_dim = 0

        for col_name, cardinality in cat_cardinalities.items():
            if embedding_dims and col_name in embedding_dims:
                dim = embedding_dims[col_name]
            else:
                dim = min(50, max(4, int(cardinality**0.5) * 2))
            self.embeddings.append(nn.Embedding(cardinality + 2, dim, padding_idx=0))
            total_embed_dim += dim

        self.cont_norm = nn.LayerNorm(num_continuous_features)
        total_input_dim = total_embed_dim + num_continuous_features

        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_block1 = TabularResidualBlock(
            hidden_dims[0], dropout_rate=dropout_rate
        )
        self.down_proj = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_block2 = TabularResidualBlock(
            hidden_dims[1], dropout_rate=dropout_rate * 0.75
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dims[2], 1),
        )

    def forward(self, x_cont, x_cat):
        embedded = [
            emb(torch.clamp(x_cat[:, i] + 1, min=0, max=emb.num_embeddings - 1))
            for i, emb in enumerate(self.embeddings)
        ]
        x_emb = (
            torch.cat(embedded, dim=1)
            if embedded
            else torch.empty(x_cont.size(0), 0, device=x_cont.device)
        )
        x_norm = self.cont_norm(x_cont)
        x = torch.cat([x_norm, x_emb], dim=1)
        x = self.input_proj(x)
        x = self.res_block1(x)
        x = self.down_proj(x)
        x = self.res_block2(x)
        out = self.head(x)
        return out.squeeze(-1)


def get_torch_components(model, lr=1e-3, weight_decay=1e-4, epochs=10):
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    return criterion, optimizer, scheduler


# ==============================================================================
# 4. DATA PROCESSING & FEATURE ENGINEERING
# ==============================================================================
print("Starting data processing and feature engineering...")

train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.csv")

train_df = pd.read_parquet(train_path)
test_df = pd.read_csv(test_path)
print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")

train_df["price"] = pd.to_numeric(train_df["price"], errors="coerce")
train_df["log10_price"] = np.log10(np.clip(train_df["price"], 1.0, None)).astype(
    np.float32
)

train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

train_df["year"] = train_df["date"].dt.year.astype(np.int16)
test_df["year"] = test_df["date"].dt.year.astype(np.int16)


# Fast vectorized categorical normalization
def clean_categoricals(df):
    for col in [
        "property_type",
        "is_new_build",
        "tenure",
        "sale_category",
        "town",
        "district",
        "county",
    ]:
        uniques = df[col].dropna().unique()
        mapping = {
            v: (
                str(v).strip().upper()
                if str(v).strip().upper() not in ["NAN", "NONE", ""]
                else "UNKNOWN"
            )
            for v in uniques
        }
        df[col] = df[col].map(mapping).fillna("UNKNOWN")
    return df


train_df = clean_categoricals(train_df)
test_df = clean_categoricals(test_df)

# Construct interaction features
for df in [train_df, test_df]:
    df["prop_tenure"] = df["property_type"] + "_" + df["tenure"]
    df["prop_new"] = df["property_type"] + "_" + df["is_new_build"]
    df["sale_prop"] = df["sale_category"] + "_" + df["property_type"]
    df["district_prop"] = df["district"] + "_" + df["property_type"]
    df["county_prop"] = df["county"] + "_" + df["property_type"]
    df["town_prop"] = df["town"] + "_" + df["property_type"]
    df["loc_triple"] = df["town"] + "||" + df["district"] + "||" + df["county"]

# Extract calendar & continuous trend features
base_date = pd.Timestamp("1995-01-01")
for df in [train_df, test_df]:
    dt = df["date"]
    df["month"] = dt.dt.month.astype(np.int8)
    df["quarter"] = dt.dt.quarter.astype(np.int8)
    df["day_of_month"] = dt.dt.day.astype(np.int8)
    df["day_of_week"] = dt.dt.dayofweek.astype(np.int8)
    df["day_of_year"] = dt.dt.dayofyear.astype(np.int16)
    df["is_friday"] = (dt.dt.dayofweek == 4).astype(np.int8)
    df["is_month_end"] = dt.dt.is_month_end.astype(np.int8)
    df["days_since_start"] = (dt - base_date).dt.days.astype(np.int32)
    df["time_trend"] = (df["year"] + (df["day_of_year"] - 1.0) / 365.25).astype(
        np.float32
    )

# Global Categorical Encodings (Consistent across train and test)
cat_cols = [
    "property_type",
    "is_new_build",
    "tenure",
    "sale_category",
    "prop_tenure",
    "prop_new",
    "sale_prop",
    "county",
    "district",
    "town",
    "district_prop",
    "county_prop",
    "town_prop",
    "loc_triple",
]

encoding_maps = {}
for col in cat_cols:
    unique_vals = sorted(train_df[col].unique().tolist())
    val_to_id = {v: i for i, v in enumerate(unique_vals)}
    encoding_maps[col] = val_to_id
    train_df[f"{col}_code"] = train_df[col].map(val_to_id).fillna(-1).astype(np.int32)
    test_df[f"{col}_code"] = test_df[col].map(val_to_id).fillna(-1).astype(np.int32)

# Frequency encoding computed on reference period (2014-2016)
ref_mask_freq = (train_df["year"] >= 2014) & (train_df["year"] <= 2016)
ref_df_freq = train_df[ref_mask_freq]

for col in ["district", "town", "county", "district_prop", "town_prop"]:
    counts = ref_df_freq[col].value_counts().to_dict()
    train_df[f"freq_log_{col}"] = np.log1p(train_df[col].map(counts).fillna(0)).astype(
        np.float32
    )
    test_df[f"freq_log_{col}"] = np.log1p(test_df[col].map(counts).fillna(0)).astype(
        np.float32
    )

del ref_df_freq
gc.collect()

# Hierarchical Empirical Bayes Target Encoding (Prior 2-Year Window)
print("Computing 2-year lagged Hierarchical Empirical Bayes Target Encodings with Momentum...")

years_to_process = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
te_records_train = []
te_records_test = None

for eval_year in years_to_process:
    ref_y_start = eval_year - 2
    ref_y_end = eval_year - 1

    hist_mask = (train_df["year"] >= ref_y_start) & (train_df["year"] <= ref_y_end)
    hist_slice = train_df.loc[
        hist_mask,
        [
            "log10_price",
            "county",
            "district",
            "town",
            "prop_tenure",
            "prop_new",
            "district_prop",
            "town_prop",
            "year",
        ],
    ].copy()

    global_mean = float(hist_slice["log10_price"].mean())

    hist_start = hist_slice.loc[hist_slice["year"] == ref_y_start]
    hist_end = hist_slice.loc[hist_slice["year"] == ref_y_end]

    m_start = hist_start["log10_price"].mean()
    m_end = hist_end["log10_price"].mean()
    momentum_national = (
        float(m_end - m_start) if (pd.notna(m_start) and pd.notna(m_end)) else 0.0
    )
    momentum_national = float(np.clip(momentum_national, -0.15, 0.15))

    # Regional Price Momentum Estimation
    c_start = hist_start.groupby("county")["log10_price"].agg(count="count", mean="mean")
    c_end = hist_end.groupby("county")["log10_price"].agg(count="count", mean="mean")
    c_growth = c_end.join(c_start, lsuffix="_end", rsuffix="_start")
    c_eff = np.minimum(c_growth["count_end"].fillna(0), c_growth["count_start"].fillna(0))
    c_raw_delta = (c_growth["mean_end"] - c_growth["mean_start"]).fillna(0.0)
    c_mom = (c_eff * c_raw_delta + 50.0 * momentum_national) / (c_eff + 50.0)
    county_momentum_map = c_mom.fillna(momentum_national).clip(-0.20, 0.20).to_dict()

    d_start = hist_start.groupby("district")["log10_price"].agg(count="count", mean="mean")
    d_end = hist_end.groupby("district")["log10_price"].agg(count="count", mean="mean")
    d_growth = d_end.join(d_start, lsuffix="_end", rsuffix="_start")
    d_eff = np.minimum(d_growth["count_end"].fillna(0), d_growth["count_start"].fillna(0))
    d_raw_delta = (d_growth["mean_end"] - d_growth["mean_start"]).fillna(0.0)
    dist_to_county = hist_slice.groupby("district")["county"].first().to_dict()
    d_prior_mom = (
        d_growth.index.to_series()
        .map(dist_to_county)
        .map(county_momentum_map)
        .fillna(momentum_national)
    )
    d_mom = (d_eff * d_raw_delta + 30.0 * d_prior_mom) / (d_eff + 30.0)
    dist_momentum_map = d_mom.fillna(d_prior_mom).clip(-0.25, 0.25).to_dict()

    # 1. County prior: smoothed towards global_mean (weight = 50)
    county_grp = (
        hist_slice.groupby("county")["log10_price"].agg(["count", "mean"]).reset_index()
    )
    county_grp["te_county"] = (
        county_grp["count"] * county_grp["mean"] + 50.0 * global_mean
    ) / (county_grp["count"] + 50.0)
    county_map = dict(zip(county_grp["county"], county_grp["te_county"]))

    # 2. District prior: smoothed towards county prior (weight = 25)
    hist_slice["c_prior"] = hist_slice["county"].map(county_map).fillna(global_mean)
    dist_grp = (
        hist_slice.groupby("district")
        .agg(
            count=("log10_price", "count"),
            mean=("log10_price", "mean"),
            c_prior=("c_prior", "mean"),
        )
        .reset_index()
    )
    dist_grp["te_district"] = (
        dist_grp["count"] * dist_grp["mean"] + 25.0 * dist_grp["c_prior"]
    ) / (dist_grp["count"] + 25.0)
    dist_map = dict(zip(dist_grp["district"], dist_grp["te_district"]))

    # 3. Town prior: smoothed towards district prior (weight = 20)
    hist_slice["d_prior"] = (
        hist_slice["district"].map(dist_map).fillna(hist_slice["c_prior"])
    )
    town_grp = (
        hist_slice.groupby("town")
        .agg(
            count=("log10_price", "count"),
            mean=("log10_price", "mean"),
            d_prior=("d_prior", "mean"),
        )
        .reset_index()
    )
    town_grp["te_town"] = (
        town_grp["count"] * town_grp["mean"] + 20.0 * town_grp["d_prior"]
    ) / (town_grp["count"] + 20.0)
    town_map = dict(zip(town_grp["town"], town_grp["te_town"]))

    # 4. District_Prop: smoothed towards district prior (weight = 15)
    dp_grp = (
        hist_slice.groupby("district_prop")
        .agg(
            count=("log10_price", "count"),
            mean=("log10_price", "mean"),
            d_prior=("d_prior", "mean"),
        )
        .reset_index()
    )
    dp_grp["te_district_prop"] = (
        dp_grp["count"] * dp_grp["mean"] + 15.0 * dp_grp["d_prior"]
    ) / (dp_grp["count"] + 15.0)
    dp_map = dict(zip(dp_grp["district_prop"], dp_grp["te_district_prop"]))

    # 5. Town_Prop: smoothed towards town prior (weight = 10)
    hist_slice["t_prior"] = (
        hist_slice["town"].map(town_map).fillna(hist_slice["d_prior"])
    )
    tp_grp = (
        hist_slice.groupby("town_prop")
        .agg(
            count=("log10_price", "count"),
            mean=("log10_price", "mean"),
            t_prior=("t_prior", "mean"),
        )
        .reset_index()
    )
    tp_grp["te_town_prop"] = (
        tp_grp["count"] * tp_grp["mean"] + 10.0 * tp_grp["t_prior"]
    ) / (tp_grp["count"] + 10.0)
    tp_map = dict(zip(tp_grp["town_prop"], tp_grp["te_town_prop"]))

    # 6. Property-type interactions (prop_tenure, prop_new)
    pt_grp = (
        hist_slice.groupby("prop_tenure")["log10_price"]
        .agg(["count", "mean"])
        .reset_index()
    )
    pt_grp["te_prop_tenure"] = (
        pt_grp["count"] * pt_grp["mean"] + 30.0 * global_mean
    ) / (pt_grp["count"] + 30.0)
    pt_map = dict(zip(pt_grp["prop_tenure"], pt_grp["te_prop_tenure"]))

    pn_grp = (
        hist_slice.groupby("prop_new")["log10_price"]
        .agg(["count", "mean"])
        .reset_index()
    )
    pn_grp["te_prop_new"] = (
        pn_grp["count"] * pn_grp["mean"] + 30.0 * global_mean
    ) / (pn_grp["count"] + 30.0)
    pn_map = dict(zip(pn_grp["prop_new"], pn_grp["te_prop_new"]))

    if eval_year == 2017:
        target_df = test_df
    else:
        target_df = train_df.loc[train_df["year"] == eval_year]

    te_c = target_df["county"].map(county_map).fillna(global_mean).astype(np.float32)
    te_d = target_df["district"].map(dist_map).fillna(te_c).astype(np.float32)
    te_t = target_df["town"].map(town_map).fillna(te_d).astype(np.float32)
    te_dp = target_df["district_prop"].map(dp_map).fillna(te_d).astype(np.float32)
    te_tp = target_df["town_prop"].map(tp_map).fillna(te_t).astype(np.float32)
    te_pt = target_df["prop_tenure"].map(pt_map).fillna(global_mean).astype(np.float32)
    te_pn = target_df["prop_new"].map(pn_map).fillna(global_mean).astype(np.float32)

    dt_diff = (target_df["time_trend"] - float(ref_y_end)).astype(np.float32)
    mom_c = target_df["county"].map(county_momentum_map).fillna(momentum_national).astype(np.float32)
    mom_d = target_df["district"].map(dist_momentum_map).fillna(mom_c).astype(np.float32)

    te_block = pd.DataFrame(
        {
            "id": target_df["id"].values,
            "hist_mean_national": np.float32(global_mean),
            "hist_trend_national": np.float32(momentum_national),
            "momentum_county": mom_c.values,
            "momentum_district": mom_d.values,
            "te_county": te_c.values,
            "te_district": te_d.values,
            "te_town": te_t.values,
            "te_district_prop": te_dp.values,
            "te_town_prop": te_tp.values,
            "te_prop_tenure": te_pt.values,
            "te_prop_new": te_pn.values,
            "te_national_projected": (global_mean + momentum_national * dt_diff).values,
            "te_county_projected": (te_c + mom_c * dt_diff).values,
            "te_district_projected": (te_d + mom_d * dt_diff).values,
            "te_town_projected": (te_t + mom_d * dt_diff).values,
            "te_district_prop_projected": (te_dp + mom_d * dt_diff).values,
            "te_town_prop_projected": (te_tp + mom_d * dt_diff).values,
            "rel_premium_district": (te_d - global_mean).astype(np.float32).values,
            "rel_premium_town": (te_t - te_d).astype(np.float32).values,
            "rel_premium_district_prop": (te_dp - te_d).astype(np.float32).values,
            "rel_premium_town_prop": (te_tp - te_t).astype(np.float32).values,
            "rel_premium_prop_tenure": (te_pt - global_mean).astype(np.float32).values,
            "rel_premium_prop_new": (te_pn - global_mean).astype(np.float32).values,
        }
    )

    if eval_year == 2017:
        te_records_test = te_block
    else:
        te_records_train.append(te_block)

te_df_train = pd.concat(te_records_train, ignore_index=True)
del te_records_train
gc.collect()

# Filter train to modern market regime (2011-2016) and merge TE features
print("Filtering training set to modern regime (2011-2016)...")
train_modern = train_df[(train_df["year"] >= 2011) & (train_df["year"] <= 2016)].copy()
del train_df
gc.collect()

train_modern = train_modern.merge(te_df_train, on="id", how="inner")
del te_df_train
gc.collect()

test_df = test_df.merge(te_records_test, on="id", how="inner")
del te_records_test
gc.collect()

# Define validation splits and clean train mask
train_modern["is_val_2016"] = (train_modern["year"] == 2016).astype(bool)
train_modern["is_val_h2_2016"] = (train_modern["date"] >= "2016-07-01").astype(bool)
train_modern["is_clean_train"] = (train_modern["price"] >= 1000) & (
    train_modern["price"] <= 50000000
)

feature_columns = [
    # Continuous Time & Trend
    "time_trend",
    "days_since_start",
    "year",
    "month",
    "quarter",
    "day_of_month",
    "day_of_week",
    "day_of_year",
    "is_friday",
    "is_month_end",
    # Categorical Encoded IDs
    "property_type_code",
    "is_new_build_code",
    "tenure_code",
    "sale_category_code",
    "prop_tenure_code",
    "prop_new_code",
    "sale_prop_code",
    "county_code",
    "district_code",
    "town_code",
    "district_prop_code",
    # Frequency / Volume Densities
    "freq_log_district",
    "freq_log_town",
    "freq_log_county",
    "freq_log_district_prop",
    "freq_log_town_prop",
    # Hierarchical Empirical Bayes Target Encodings & Momentum
    "hist_mean_national",
    "hist_trend_national",
    "momentum_county",
    "momentum_district",
    "te_county",
    "te_district",
    "te_town",
    "te_district_prop",
    "te_town_prop",
    "te_prop_tenure",
    "te_prop_new",
    "te_national_projected",
    "te_county_projected",
    "te_district_projected",
    "te_town_projected",
    "te_district_prop_projected",
    "te_town_prop_projected",
    # Relative Price Premiums
    "rel_premium_district",
    "rel_premium_town",
    "rel_premium_district_prop",
    "rel_premium_town_prop",
    "rel_premium_prop_tenure",
    "rel_premium_prop_new",
]

categorical_columns = [
    "property_type_code",
    "is_new_build_code",
    "tenure_code",
    "sale_category_code",
    "prop_tenure_code",
    "prop_new_code",
    "sale_prop_code",
    "county_code",
    "district_code",
    "town_code",
    "district_prop_code",
]

cont_cols = [c for c in feature_columns if c not in categorical_columns]

# Save metadata for reproducibility
meta_dict = {
    "feature_columns": feature_columns,
    "categorical_columns": categorical_columns,
    "target_column": "log10_price",
    "train_rows": len(train_modern),
    "test_rows": len(test_df),
    "num_features": len(feature_columns),
    "val_2016_rows": int(train_modern["is_val_2016"].sum()),
}
with open(os.path.join(WORKING_DIR, "feature_meta.json"), "w") as f:
    json.dump(meta_dict, f, indent=2)

# ==============================================================================
# 5. MODEL ARCHITECTURE VERIFICATION
# ==============================================================================
cat_cardinalities = {
    "property_type_code": 8,
    "is_new_build_code": 4,
    "tenure_code": 5,
    "sale_category_code": 4,
    "prop_tenure_code": 20,
    "prop_new_code": 15,
    "sale_prop_code": 15,
    "county_code": 200,
    "district_code": 600,
    "town_code": 2000,
    "district_prop_code": 3000,
}

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_model = PropertyPriceTabularNet(
    num_continuous_features=len(cont_cols),
    cat_cardinalities=cat_cardinalities,
    hidden_dims=(256, 256, 128),
    dropout_rate=0.2,
).to(torch_device)

criterion, optimizer, scheduler = get_torch_components(torch_model, lr=1e-3, epochs=10)

# Quick gradient verification on dummy batch
dummy_cont = torch.randn(64, len(cont_cols), device=torch_device)
dummy_cat = torch.zeros(
    64, len(categorical_columns), dtype=torch.long, device=torch_device
)
dummy_target = torch.tensor([5.3] * 64, dtype=torch.float32, device=torch_device)

torch_model.train()
dummy_pred = torch_model(dummy_cont, dummy_cat)
dummy_loss = criterion(dummy_pred, dummy_target)
optimizer.zero_grad()
dummy_loss.backward()
optimizer.step()
scheduler.step()

del dummy_cont, dummy_cat, dummy_target, dummy_pred, dummy_loss, torch_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==============================================================================
# 6. TRAINING & OUT-OF-TIME VALIDATION
# ==============================================================================
# Exponential recency sample weighting
train_modern["sample_weight"] = np.exp(
    0.20 * (train_modern["time_trend"] - 2011.0)
).astype(np.float32)

val_mask = train_modern["is_val_2016"].values
train_mask = (~val_mask) & (train_modern["is_clean_train"].values)

X_train = train_modern.loc[train_mask, feature_columns]
y_train = train_modern.loc[train_mask, "log10_price"].values
w_train = train_modern.loc[train_mask, "sample_weight"].values

X_val = train_modern.loc[val_mask, feature_columns]
y_val = train_modern.loc[val_mask, "log10_price"].values
val_true_price = train_modern.loc[val_mask, "price"].values

print(f"Validation split: {len(X_train):,} train samples, {len(X_val):,} val samples.")

val_model = create_lgb_model()
callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

val_model.fit(
    X_train,
    y_train,
    sample_weight=w_train,
    eval_set=[(X_val, y_val)],
    categorical_feature=categorical_columns,
    callbacks=callbacks,
)

best_iteration = (
    val_model.best_iteration_
    if hasattr(val_model, "best_iteration_") and val_model.best_iteration_ > 0
    else 1500
)
print(f"Validation model training converged at iteration: {best_iteration}")

best_model_path = os.path.join(WORKING_DIR, "best_lgb_model.txt")
val_model.booster_.save_model(best_model_path)

val_pred_log10 = val_model.predict(X_val)
val_pred_price = np.clip(10.0**val_pred_log10, 1.0, None)
final_val_score = compute_raw_price_rmse(val_true_price, val_pred_price)

del X_train, y_train, w_train, X_val, y_val, val_true_price, val_pred_log10, val_pred_price
gc.collect()

# ==============================================================================
# 7. PRODUCTION MODEL RETRAINING & INFERENCE
# ==============================================================================
full_estimators = max(500, int(best_iteration * 1.10))
print(
    f"Retraining production model on full modern history (2011-2016) with {full_estimators} estimators..."
)

full_clean_mask = train_modern["is_clean_train"].values
X_full = train_modern.loc[full_clean_mask, feature_columns]
y_full = train_modern.loc[full_clean_mask, "log10_price"].values
w_full = train_modern.loc[full_clean_mask, "sample_weight"].values

prod_model = create_lgb_model()
prod_model.set_params(n_estimators=full_estimators)
prod_model.fit(
    X_full,
    y_full,
    sample_weight=w_full,
    categorical_feature=categorical_columns,
)

prod_model_path = os.path.join(WORKING_DIR, "prod_lgb_model.txt")
prod_model.booster_.save_model(prod_model_path)

del X_full, y_full, w_full, train_modern
gc.collect()

print("Generating test predictions on 2017 holdout transactions...")
X_test = test_df[feature_columns]
if X_test.isna().any().any():
    X_test = X_test.fillna(0)

test_pred_log10 = prod_model.predict(X_test)
test_pred_price = np.clip(10.0**test_pred_log10, 1.0, None)

# ==============================================================================
# 8. SUBMISSION GENERATION & VERIFICATION
# ==============================================================================
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

submission_raw = pd.DataFrame(
    {"id": test_df["id"].values, "price": np.round(test_pred_price, 2)}
)

submission = sample_sub[["id"]].merge(submission_raw, on="id", how="left")

assert len(submission) == len(
    sample_sub
), f"Row count mismatch: {len(submission)} vs expected {len(sample_sub)}"
assert not submission["price"].isna().any(), "NaN values detected in final predictions!"
assert (submission["price"] >= 1.0).all(), "Predictions contain values < 1.0!"

final_sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission.to_csv(final_sub_path, index=False)
print(f"Saved submission with {len(submission)} rows to {final_sub_path}")

print(f"Final Validation Score: {final_val_score}")
