"""Round 2: verify soil/wilderness are exactly-one-hot (collapsible to a code)."""
import pandas as pd
from common import WS_ROOT

df = pd.read_csv(WS_ROOT / "input" / "train.csv", nrows=500_000)
soil = [c for c in df.columns if c.startswith("Soil_Type")]
wild = [c for c in df.columns if c.startswith("Wilderness_Area")]

s = df[soil].sum(axis=1)
w = df[wild].sum(axis=1)
print("Soil active-count distribution:\n", s.value_counts())
print("\nWilderness active-count distribution:\n", w.value_counts())
print("\nn distinct soil types present:", (df[soil].sum() > 0).sum(), "/", len(soil))
print("n distinct wilderness present:", (df[wild].sum() > 0).sum(), "/", len(wild))
