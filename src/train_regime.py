"""
train_regime.py — Phase 9: HMM Regime Conditioning
==================================================
- Goal: Categorize the market into 3 "Hidden States" (e.g. Calm, Trending, Crisis).
- Mechanism: GaussianHMM (hmmlearn)
- Rule: Fit only on Train, Predict on Train & Test (Prevent Leakage).
- Result: Updated parquet files with 'HMM_Regime' column.
"""

import polars as pl
import numpy as np
import yaml
import joblib
import os
from hmmlearn import hmm

# Config
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

ASSETS = ["nifty", "gold", "usdinr"]
REGIME_FEATS = config.get("Regime_Features")



REPORTS = []

# Function: Train & Inject
def process_asset_regime(asset):
    msg_header = f"\n{'='*60}\n  ASSET: {asset.upper()} (GaussianHMM)\n{'='*60}"
    print(msg_header)
    REPORTS.append(msg_header)

    train_path = f"data/processed/train/{asset}.parquet"
    test_path  = f"data/processed/test/{asset}.parquet"

    if not os.path.exists(train_path):
        print(f"   [SKIP] Train data missing at {train_path}")
        return

    train = pl.read_parquet(train_path)
    test  = pl.read_parquet(test_path)

    X_train_raw = train.select(REGIME_FEATS).to_numpy()
    X_test_raw  = test.select(REGIME_FEATS).to_numpy()

    # Fit only on Train
    model = hmm.GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    
    fit_msg = f"   Fitting HMM on {len(X_train_raw)} sample points..."
    print(fit_msg)
    REPORTS.append(fit_msg)
    
    model.fit(X_train_raw)

    # Predict on both
    train_states = model.predict(X_train_raw)
    test_states  = model.predict(X_test_raw)

    # Interpret States
    vol_idx = REGIME_FEATS.index("Vol_20d") if "Vol_20d" in REGIME_FEATS else 1
    ret_idx = REGIME_FEATS.index("Ret_5d") if "Ret_5d" in REGIME_FEATS else 0

    interp_header = "\n   --- State Interpretation Matrix ---"
    print(interp_header)
    REPORTS.append(interp_header)

    for s in range(3):
        mask = train_states == s
        s_vol = X_train_raw[mask, vol_idx].mean() if mask.any() else 0
        s_ret = X_train_raw[mask, ret_idx].mean() if mask.any() else 0
        count = mask.sum()
        pct   = (count / len(train)) * 100
        
        if s_vol > X_train_raw[:, vol_idx].mean() * 1.25:
            label = "CRISIS (High Vol)"
        elif s_vol < X_train_raw[:, vol_idx].mean() * 0.75:
            label = "CALM (Low Vol)"
        else:
            label = "TRENDING (Normal)"
            
        row = f"   State {s}: {label:20} | Vol_20d={s_vol:.4f} | Ret_5d={s_ret:.4f} | Dist={pct:.1f}%"
        print(row)
        REPORTS.append(row)

    # Save Model
    model_file = f"models/regime/{asset}_hmm.joblib"
    joblib.dump(model, model_file)
    
    save_msg = f"\n   [DONE] Saved HMM Model  -> {model_file}"
    print(save_msg)
    REPORTS.append(save_msg)

# Main 
for asset in ASSETS:
    process_asset_regime(asset)

# Finalize Report
with open("reports/regime_results.txt", "w") as f:
    f.write("\n".join(REPORTS))

print("\nPhase 9: Regime Conditioning Complete. Report saved to reports/regime_results.txt")
