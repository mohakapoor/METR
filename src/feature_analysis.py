"""
feature_analysis.py — Phase 8: Meta-Filter Feature Analysis
===========================================================
Loads each asset's Phase 7 XGBoost meta-model and test data to generate:
  1. Feature Importance (Gain)
  2. Feature vs Meta_Label Correlation (Win: 1 vs Loss: 0)
  3. Combined plot per asset

Reports saved to: reports/feature_analysis/
"""

import os
import polars as pl
import numpy as np
import joblib
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Config 
with open("config.yaml") as f:
    config = yaml.safe_load(f)

EXP_FEATURES   = config["Exposure_Features"]
CROSS_FEATURES = config["Cross_Asset_Features"]

ASSETS = ["nifty", "gold", "usdinr"]
REPORT_DIR = "reports/feature_analysis"
os.makedirs(REPORT_DIR, exist_ok=True)

# Per-Asset Analysis
for asset in ASSETS:
    model_path = f"models/meta/{asset}_xgb_meta.joblib"
    if not os.path.exists(model_path):
        print(f"[SKIP] {asset} — meta-model not found at {model_path}")
        continue

    model = joblib.load(model_path)
    
    
    test = pl.read_parquet(f"data/processed/test/{asset}.parquet")

    features  = EXP_FEATURES + CROSS_FEATURES.get(asset, [])
    available = list(dict.fromkeys(f for f in features if f in test.columns))  

    # Meta-Labels: 1 = Win, 0 = Loss
    labels = test["Meta_Label"].to_numpy() 
    X = test.select(available).to_pandas()

    print(f"\n{'='*60}")
    print(f"  {asset.upper()} | {len(available)} features | {len(labels)} test rows")
    print(f"{'='*60}")

    # 1. Feature Importance (Gain)
    # Handle CalibratedClassifierCV wrapper vs raw XGBoost
    if hasattr(model, "calibrated_classifiers_"):
        # Extract from the first estimator in the calibration ensemble
        # (Assuming cv='prefit' so they are identical)
        actual_booster = model.calibrated_classifiers_[0].estimator.get_booster()
    else:
        actual_booster = model.get_booster()

    scores = actual_booster.get_score(importance_type="gain")
    feat_order = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    feat_names = [f[0] for f in feat_order]
    feat_vals  = [f[1] for f in feat_order]

    print(f"\n  Feature Importance (Gain):")
    max_gain = max(feat_vals) if feat_vals else 1
    for name, val in feat_order[:15]: # Show top 15
        bar = "█" * int(val / max_gain * 30)
        print(f"    {name:25s} {val:8.2f}  {bar}")

    #  2. Correlation with Meta_Label (Win:1 vs Loss:0) 
    corrs = {}
    for feat in available:
        col = X[feat].values.astype(float)
        mask = ~np.isnan(col)
        if mask.sum() > 10:
            corrs[feat] = np.corrcoef(col[mask], labels[mask])[0, 1]
        else:
            corrs[feat] = 0.0

    print(f"\n  Correlation with Win (Meta_Label=1):")
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, r in sorted_corrs[:10]:
        print(f"    {feat:25s} {r:+8.4f}")

    # 3. Plot 
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(available) * 0.35)))
    fig.suptitle(f"Phase 8 Analysis — {asset.upper()} (Meta-Filter)", fontsize=14, fontweight="bold")

    # Left: Gain importance
    ax = axes[0]
    colors = ["#4CAF50" if v > np.median(feat_vals) else "#90CAF9" for v in feat_vals[:len(available)][::-1]]
    ax.barh(feat_names[:len(available)][::-1], feat_vals[:len(available)][::-1], color=colors)
    ax.set_title("Alpha Source: Feature Importance (Gain)", fontweight="bold")
    ax.set_xlabel("Gain")

    # Right: Correlation with Meta_Label
    ax = axes[1]
    sorted_feats = [f[0] for f in sorted_corrs]
    sorted_vals  = [f[1] for f in sorted_corrs]
    
    colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in sorted_vals[::-1]]
    ax.barh(sorted_feats[::-1], sorted_vals[::-1], color=colors)
    ax.set_title("Alpha Direction: Correlation with Win", fontweight="bold")
    ax.set_xlabel("Correlation coefficient (r)")
    ax.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    out = f"{REPORT_DIR}/{asset}_meta_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ Saved → {out}")

    # Append to summary file
    summary_path = f"{REPORT_DIR}/summary.txt"
    with open(summary_path, "a") as sf:
        sf.write(f"\nASSET: {asset.upper()}\n")
        sf.write(f"{'-'*30}\n")
        sf.write("Top 5 by Gain (Importance):\n")
        for name, val in feat_order[:5]:
            sf.write(f"  - {name:20s}: {val:.2f}\n")
        sf.write("\nTop 5 by Correlation ( r ):\n")
        for name, val in sorted_corrs[:5]:
            sf.write(f"  - {name:20s}: {val:.4f}\n")
        sf.write("\n")

print("\nPhase 8 analysis complete.")
