# Data-Driven Cyber-Attack Detection in Networked Control Systems

> **False Data Injection (FDI) attack detection on the SWaT water treatment testbed using an LSTM Autoencoder — F1: 0.9928 | Recall: 1.0 | Detection Latency: 1 step**

---

## Overview

Modern Industrial Control Systems (ICS) are vulnerable to False Data Injection attacks, where an adversary manipulates sensor readings to cause unsafe physical actions while evading rule-based detectors.

This project builds a **purely data-driven anomaly detection system** that:
- Learns the physical correlations between 52 sensors during normal operation
- Flags timesteps where sensor readings violate those learned correlations
- Identifies *which* sensors are responsible for the anomaly (interpretability)
- Serves predictions in real time via a FastAPI inference server

---

## Results

| Metric | This Project | Literature Baseline |
|---|---|---|
| Precision (Attack) | **0.9857** | 0.99 |
| Recall (Attack) | **1.0000** | 0.62 |
| F1 Score | **0.9928** | 0.77 |
| ROC-AUC | **0.9010** | — |
| Detection Latency | **1 step (1s)** | — |
| Missed Attacks | **0 / 36** | — |

---

## Architecture

```
Input X(t) ∈ R^(30×52)
        ↓
  Encoder: LSTM(64) → LSTM(32)
        ↓
  Bottleneck: z ∈ R^32
        ↓
  Decoder: LSTM(32) → LSTM(64) → Dense(52)
        ↓
Output X_hat(t) ∈ R^(30×52)
        ↓
  Anomaly Score: s(t) = mean MSE per sensor
  Flag if s(t) ≥ τ (99th percentile of normal val errors)
```

---

## Dataset

Uses the [SWaT Dataset](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system) from Kaggle.

Download and place files in `data/`:
```
data/
├── normal.csv     ← SWaT_Dataset_Normal_v1.csv
└── attack.csv     ← SWaT_Dataset_Attack_v0.csv
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/swat-fdi-detection
cd swat-fdi-detection

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

---

## Run

```bash
# 1. Build mixed test set
python build_test_set.py

# 2. Train LSTM Autoencoder (GPU recommended)
python train.py --normal data/normal.csv --attack data/attack.csv \
                --arch lstm --window 30 --latent 32 --batch 256 --epochs 100

# 3. Fit detection threshold
python threshold.py --normal data/normal.csv --attack data/test_mixed.csv \
                    --percentile 99

# 4. Evaluate
python evaluate.py --normal data/normal.csv --attack data/test_mixed.csv

# 5. Start inference server
uvicorn infer:app --host 0.0.0.0 --port 8000

# 6. Stream live sensor data
python test_inference.py
```

---

## Project Structure

| File | Purpose |
|---|---|
| `dataset.py` | Data loading, scaling, sliding window dataset |
| `model.py` | LSTM Autoencoder + CNN Autoencoder architectures |
| `train.py` | Training loop with early stopping |
| `threshold.py` | Threshold fitting and F1 sweep |
| `evaluate.py` | Full evaluation + sensor heatmap |
| `infer.py` | FastAPI real-time inference server |
| `build_test_set.py` | Creates mixed normal+attack test set |
| `test_inference.py` | Simulates live sensor stream to server |

---

## Tech Stack

`PyTorch` · `FastAPI` · `scikit-learn` · `pandas` · `matplotlib` · `python-pptx`

---

## Author

**Varun Kulkarni**  
University of Texas at Arlington  
ID: 1002228294