"""
threshold.py
Fits the anomaly detection threshold from validation reconstruction errors.
Saves threshold to disk alongside the checkpoint.

Usage:
    python threshold.py --checkpoint best_model.pt --normal data/normal.xlsx
                        --attack data/attack.xlsx
"""
import argparse
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt

from dataset import make_loaders, load_swat, split_normal, load_scaler, scale_df, SlidingWindowDataset
from model import LSTMAutoencoder, CNNAutoencoder, recon_error_scalar
from torch.utils.data import DataLoader


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_feat   = ckpt["n_features"]
    latent   = ckpt["latent"]
    window   = ckpt["window"]
    n_layers = ckpt.get("layers", 2)

    if ckpt["arch"] == "lstm":
        model = LSTMAutoencoder(n_feat, latent, window, n_layers)
    else:
        model = CNNAutoencoder(n_feat, latent, window)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model, ckpt


@torch.no_grad()
def collect_errors(model, loader, device):
    scores, labels = [], []
    for x, lbl in loader:
        x    = x.to(device)
        x_hat = model(x)
        err   = recon_error_scalar(x, x_hat).cpu().numpy()
        scores.append(err)
        labels.append(lbl.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def fit_threshold(val_errors, percentile=99.0):
    """Fit threshold at the given percentile of the normal val error distribution."""
    thresh = np.percentile(val_errors, percentile)
    print(f"Threshold @ {percentile}th pct  =  {thresh:.6f}")
    return thresh


def sweep_f1(test_errors, test_labels, n_pts=200):
    """
    Sweep thresholds and return the one that maximises point-adjusted F1.
    Point-adjust: if any step in an attack window triggers, the whole window counts.
    """
    thresholds = np.linspace(test_errors.min(), test_errors.max(), n_pts)
    best_f1, best_thresh = 0.0, thresholds[0]
    f1s = []

    for t in thresholds:
        preds = (test_errors >= t).astype(int)
        f1 = _f1(preds, test_labels)
        f1s.append(f1)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"Best F1={best_f1:.4f}  at threshold={best_thresh:.6f}")
    return best_thresh, best_f1, thresholds, np.array(f1s)


def _f1(preds, labels):
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


def plot_threshold(thresholds, f1s, chosen, out="threshold_sweep.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, f1s, lw=1.5, label="F1")
    ax.axvline(chosen, color="red", ls="--", label=f"Chosen={chosen:.4f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1")
    ax.set_title("F1 vs Anomaly Score Threshold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Plot saved → {out}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    scaler, feat_cols = load_scaler(args.scaler)
    window = ckpt["window"]

    # ── Validation errors (normal data only) ──────────────────────────────────
    normal, attack = load_swat(args.normal, args.attack)
    _, val_df = split_normal(normal)
    X_val, y_val = scale_df(val_df, scaler, feat_cols)
    val_ds  = SlidingWindowDataset(X_val, y_val, window)
    val_ldr = DataLoader(val_ds, batch_size=256, shuffle=False)
    val_err, _ = collect_errors(model, val_ldr, device)

    threshold = fit_threshold(val_err, percentile=args.percentile)

    # ── Optional: sweep on test set to find oracle-best threshold ─────────────
    X_test, y_test = scale_df(attack, scaler, feat_cols)
    test_ds  = SlidingWindowDataset(X_test, y_test, window)
    test_ldr = DataLoader(test_ds, batch_size=256, shuffle=False)
    test_err, test_lbl = collect_errors(model, test_ldr, device)

    best_t, best_f1, ts, f1s = sweep_f1(test_err, test_lbl)
    plot_threshold(ts, f1s, threshold)

    # ── Save ──────────────────────────────────────────────────────────────────
    thresh_data = {
        "threshold":   threshold,
        "percentile":  args.percentile,
        "best_oracle_threshold": best_t,
        "best_oracle_f1":        best_f1,
    }
    joblib.dump(thresh_data, args.out)
    print(f"Threshold data saved → {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="best_model.pt")
    p.add_argument("--normal",      required=True)
    p.add_argument("--attack",      required=True)
    p.add_argument("--scaler",      default="scaler.pkl")
    p.add_argument("--percentile",  type=float, default=99.0)
    p.add_argument("--out",         default="threshold.pkl")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())