"""Round 1 exploration: shape, dtypes, target distribution, missing, constants."""
import pandas as pd
import numpy as np
from common import WS_ROOT

path = WS_ROOT / "input" / "train.csv"

# Full target distribution is cheap (one column).
y = pd.read_csv(path, usecols=["Cover_Type"])["Cover_Type"]
print("=== n rows (full) ===", len(y))
print("\n=== Cover_Type distribution (full) ===")
print(y.value_counts().sort_index())
print("\nnormalized:")
print((y.value_counts(normalize=True).sort_index() * 100).round(3))

# A sample for column-level stats.
df = pd.read_csv(path, nrows=200_000)
print("\n=== shape (sample) ===", df.shape)
print("\n=== dtypes value_counts ===")
print(df.dtypes.value_counts())
print("\n=== missing per column (sample, nonzero only) ===")
miss = df.isna().sum()
print(miss[miss > 0] if (miss > 0).any() else "none")

print("\n=== constant / near-constant columns (sample) ===")
nun = df.nunique()
print(nun[nun <= 1] if (nun <= 1).any() else "none constant")
print("\n=== columns with <=2 unique (binary indicators) count ===",
      int((nun <= 2).sum()))

print("\n=== numeric (non-binary) feature summary ===")
nonbin = [c for c in df.columns if nun[c] > 2 and c not in ("Id", "Cover_Type")]
print(df[nonbin].describe().T[["min", "mean", "max", "std"]])

# Soil type coverage across full data would be ideal but sample is indicative.
soil = [c for c in df.columns if c.startswith("Soil_Type")]
print("\n=== Soil_Type column sums (sample) -- spot all-zero ones ===")
sums = df[soil].sum()
print("all-zero soil cols in sample:", list(sums[sums == 0].index))
