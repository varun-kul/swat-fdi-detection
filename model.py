"""
model.py
Two autoencoder architectures:
  - LSTMAutoencoder  (baseline)
  - CNNAutoencoder   (faster ablation)
Both learn to reconstruct normal windows; elevated MSE → anomaly.
"""
import torch
import torch.nn as nn


# ── LSTM Autoencoder ───────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    def __init__(self, n_features, latent_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, latent_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )

    def forward(self, x):
        # x: (B, W, F)
        _, (h, _) = self.lstm(x)   # h: (layers, B, latent)
        return h[-1]               # take last layer's hidden: (B, latent)


class LSTMDecoder(nn.Module):
    def __init__(self, n_features, latent_dim, window, n_layers=2, dropout=0.2):
        super().__init__()
        self.window = window
        self.lstm = nn.LSTM(
            latent_dim, latent_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.proj = nn.Linear(latent_dim, n_features)

    def forward(self, z):
        # z: (B, latent) → repeat → (B, W, latent)
        z_rep = z.unsqueeze(1).repeat(1, self.window, 1)
        out, _ = self.lstm(z_rep)            # (B, W, latent)
        return self.proj(out)                # (B, W, F)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, latent_dim=32, window=30, n_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, latent_dim, n_layers, dropout)
        self.decoder = LSTMDecoder(n_features, latent_dim, window, n_layers, dropout)

    def forward(self, x):
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    @property
    def name(self):
        return "LSTMAutoencoder"


# ── 1D CNN Autoencoder ─────────────────────────────────────────────────────────

class CNNAutoencoder(nn.Module):
    """
    Encoder: 3× Conv1d with stride-2 downsampling → bottleneck.
    Decoder: 3× ConvTranspose1d upsampling.
    Input shape: (B, F, W) — channels-first for Conv1d.
    """
    def __init__(self, n_features, latent_channels=32, window=30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 64,             kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(64,          latent_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 64,        kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(64,             n_features, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
        )
        self._window = window

    def forward(self, x):
        # x: (B, W, F) → transpose → (B, F, W)
        x = x.transpose(1, 2)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        # Crop/pad to original window length
        x_hat = x_hat[:, :, : self._window]
        return x_hat.transpose(1, 2)          # → (B, W, F)

    @property
    def name(self):
        return "CNNAutoencoder"


# ── Reconstruction error helpers ───────────────────────────────────────────────

def recon_error_per_sample(x, x_hat):
    """Mean squared error averaged over time steps; shape: (B, F)."""
    return ((x - x_hat) ** 2).mean(dim=1)   # (B, F)


def recon_error_scalar(x, x_hat):
    """Scalar anomaly score per sample (mean over features): (B,)."""
    return recon_error_per_sample(x, x_hat).mean(dim=-1)  # (B,)