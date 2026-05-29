"""
export_demo.py
Exports a small, pre-scaled demo dataset (demo.npz) for the deployed server.
Captures 200 normal rows just before the first attack + all attack rows.
This file is committed to the repo so Render doesn't need the raw CSVs.

Run locally:
    python export_demo.py
Then commit demo.npz to git.
"""
import warnings
warnings.filterwarnings("ignore")   # suppress FastAPI/pydantic deprecation noise

import numpy as np
import pandas as pd
import joblib

MIXED_PATH  = "data/test_mixed.csv"
SCALER_PATH = "scaler.pkl"
OUT_PATH    = "demo.npz"
N_NORMAL    = 200   # normal rows to include before first attack


df = pd.read_csv(MIXED_PATH, low_memory=False)
df.columns = df.columns.str.strip()

feat_cols = [c for c in df.columns
             if c not in {"Timestamp", "Normal/Attack"}
             and not c.startswith("Unnamed")]

df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=feat_cols).reset_index(drop=True)

df["Normal/Attack"] = (
    df["Normal/Attack"].astype(str).str.strip()
    .str.replace("!", "", regex=False).str.lower()
    .map({"normal": 0, "attack": 1}).fillna(0).astype(int)
)

labels = df["Normal/Attack"].values

# Find first attack and take N_NORMAL rows before it + all attack rows
first_atk = np.where(labels == 1)[0][0]
start_idx  = max(0, first_atk - N_NORMAL)

demo_df  = df.iloc[start_idx:].reset_index(drop=True)
demo_lbl = demo_df["Normal/Attack"].values

# Scale
scaler, _ = joblib.load(SCALER_PATH)
X_scaled  = scaler.transform(demo_df[feat_cols].values).astype(np.float32)

np.savez_compressed(OUT_PATH, X=X_scaled, labels=demo_lbl)

n_norm = (demo_lbl == 0).sum()
n_atk  = (demo_lbl == 1).sum()
print(f"Saved {OUT_PATH}")
print(f"  Normal rows : {n_norm:,}")
print(f"  Attack rows : {n_atk:,}")
print(f"  Total       : {len(demo_lbl):,}")
print(f"  File size   : {__import__('os').path.getsize(OUT_PATH)/1024:.1f} KB")