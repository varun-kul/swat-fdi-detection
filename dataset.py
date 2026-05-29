"""
dataset.py
Handles loading, preprocessing, and windowing for the Kaggle SWaT dataset.

Kaggle dataset: vishala28/swat-dataset-secure-water-treatment-system
  Files:
    SWaT_Dataset_Normal_v1.csv   (495,000 rows × 53 cols: Timestamp + 51 sensors + Normal/Attack)
    SWaT_Dataset_Attack_v0.csv   (449,919 rows × 53 cols: same structure)

51 sensors across 6 stages:
  Stage 1 (P1): FIT101, LIT101, MV101, P101, P102
  Stage 2 (P2): AIT201, AIT202, AIT203, FIT201, MV201, P201–P206
  Stage 3 (P3): DPIT301, FIT301, LIT301, MV301–MV304, P301, P302
  Stage 4 (P4): AIT401, AIT402, FIT401, LIT401, P401, P402, UV401
  Stage 5 (P5): AIT501–504, FIT501–504, P501, P502
  Stage 6 (P6): FIT601, P601–P603
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import os

# ── Column config ──────────────────────────────────────────────────────────────

LABEL_COL     = "Normal/Attack"
TIMESTAMP_COL = "Timestamp"
# Columns that are never features
_NON_FEATURE  = {LABEL_COL, TIMESTAMP_COL}

def get_feature_cols(df: pd.DataFrame):
    """
    Infer feature columns directly from the dataframe.
    Drops Timestamp, label, and any unnamed index columns.
    Works regardless of how many sensors the specific Kaggle version has.
    """
    return [c for c in df.columns
            if c not in _NON_FEATURE and not c.startswith("Unnamed")]

def get_actuator_cols(feat_cols):
    """Heuristic: columns whose name starts with P, MV, or UV are actuators."""
    return [c for c in feat_cols if c[:2] in ("MV", "UV") or c[0] == "P"]


# ── Loader ─────────────────────────────────────────────────────────────────────

def _read_swat(path: str) -> pd.DataFrame:
    """
    Read a SWaT CSV (Kaggle version).
    Handles:
      - Optional leading whitespace in column names
      - The Kaggle version's occasional 2-row header (row 0 = units, row 1 = data)
      - Label values: 'Normal', 'Attack', 'Attack!' (some versions have trailing '!')
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, header=0)
    else:
        df = pd.read_csv(path, header=0, low_memory=False)

    df.columns = df.columns.str.strip()

    # Drop unnamed index column if present
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # If the first data row is a unit/secondary-header row, drop it
    # (detect by trying to parse the first non-label, non-timestamp column as float)
    first_feat = get_feature_cols(df)
    if first_feat:
        try:
            pd.to_numeric(df.iloc[0][first_feat[0]])
        except (ValueError, TypeError):
            df = df.iloc[1:].reset_index(drop=True)

    # Cast sensor columns to float
    for c in get_feature_cols(df):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalise label
    df[LABEL_COL] = (
        df[LABEL_COL]
        .astype(str)
        .str.strip()
        .str.replace("!", "", regex=False)
        .str.lower()
        .map({"normal": 0, "attack": 1})
        .fillna(0)
        .astype(int)
    )

    df = df.dropna(subset=get_feature_cols(df)).reset_index(drop=True)
    return df


def load_swat(normal_path: str, attack_path: str):
    normal = _read_swat(normal_path)
    attack = _read_swat(attack_path)
    print(f"Normal  : {normal.shape}  |  attack rows: {normal[LABEL_COL].sum()}")
    print(f"Attack  : {attack.shape}  |  attack rows: {attack[LABEL_COL].sum()}"
          f"  ({100*attack[LABEL_COL].mean():.1f}%)")
    return normal, attack


# ── Preprocessing ──────────────────────────────────────────────────────────────

def split_normal(normal_df: pd.DataFrame, train_frac: float = 0.8):
    """Chronological 80/20 split on the normal-only dataframe."""
    cut = int(len(normal_df) * train_frac)
    return normal_df.iloc[:cut].copy(), normal_df.iloc[cut:].copy()


def fit_scaler(train_df: pd.DataFrame, feat_cols=None, scaler_path="scaler.pkl"):
    if feat_cols is None:
        feat_cols = get_feature_cols(train_df)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[feat_cols].values)
    joblib.dump((scaler, feat_cols), scaler_path)
    print(f"Scaler fitted on {len(feat_cols)} features → {scaler_path}")
    return scaler, feat_cols


def load_scaler(scaler_path="scaler.pkl"):
    return joblib.load(scaler_path)


def scale_df(df: pd.DataFrame, scaler, feat_cols):
    """Apply pre-fitted scaler. Returns (X: float32 ndarray, y: int64 ndarray)."""
    X = scaler.transform(df[feat_cols].values).astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int64) if LABEL_COL in df.columns else None
    return X, y


# ── Dataset ────────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """
    Converts a (T, F) array into overlapping windows (W, F).

    Mathematical formulation:
        X(t) = [x̃(t-W+1), ..., x̃(t)]  ∈ ℝ^{W×F}
    Label: 1 if ANY timestep in the window is an attack, else 0.
    This is consistent with point-adjusted evaluation.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray = None,
                 window: int = 30, step: int = 1):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) if y is not None else None
        self.W = window
        self.step = step
        self.indices = list(range(0, len(X) - window, step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x_win = self.X[i : i + self.W]
        lbl = int(self.y[i : i + self.W].any()) if self.y is not None else -1
        return x_win, torch.tensor(lbl, dtype=torch.long)


# ── Convenience pipeline ───────────────────────────────────────────────────────

def make_loaders(normal_path, attack_path,
                 window=30, step=1, batch=64,
                 train_frac=0.8, scaler_path="scaler.pkl"):
    """
    Full pipeline: load → split → scale → window → DataLoader.
    Returns (train_loader, val_loader, test_loader, feat_cols).
    """
    normal, attack = load_swat(normal_path, attack_path)

    # Keep only rows labeled normal in the normal file
    # (version 0 may have a few anomalous rows at the start)
    normal = normal[normal[LABEL_COL] == 0].reset_index(drop=True)

    train_df, val_df = split_normal(normal, train_frac)
    scaler, feat_cols = fit_scaler(train_df, scaler_path=scaler_path)

    X_tr, y_tr = scale_df(train_df, scaler, feat_cols)
    X_va, y_va = scale_df(val_df,   scaler, feat_cols)
    X_te, y_te = scale_df(attack,   scaler, feat_cols)

    ds_tr = SlidingWindowDataset(X_tr, y_tr, window, step)
    ds_va = SlidingWindowDataset(X_va, y_va, window, step=1)
    ds_te = SlidingWindowDataset(X_te, y_te, window, step=1)

    kw = dict(num_workers=4, pin_memory=torch.cuda.is_available())
    loaders = (
        DataLoader(ds_tr, batch_size=batch, shuffle=True,  **kw),
        DataLoader(ds_va, batch_size=batch, shuffle=False, **kw),
        DataLoader(ds_te, batch_size=batch, shuffle=False, **kw),
    )
    print(f"Windows → train: {len(ds_tr):,} | val: {len(ds_va):,} | test: {len(ds_te):,}")
    return *loaders, feat_cols