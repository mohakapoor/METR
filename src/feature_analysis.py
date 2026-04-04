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

# Per-Asset Analysis
for asset in ASSETS:
    model_path = f"models/meta/{asset}_xgb_meta.joblib"
    model = joblib.load(model_path) 
    test = pl.read_parquet(f"data/processed/test/{asset}.parquet")
    features  = EXP_FEATURES + CROSS_FEATURES.get(asset, [])
    available = list(dict.fromkeys(f for f in features if f in test.columns))  

    labels = test["Meta_Label"].to_numpy() 
    X = test.select(available).to_pandas()

    print(f"\n{'='*60}")
    print(f"  {asset.upper()} | {len(available)} features | {len(labels)} test rows")
    print(f"{'='*60}")

    # 1. Feature Importance (Gain)
    if hasattr(model, "calibrated_classifiers_"):
        actual_booster = model.calibrated_classifiers_[0].estimator.get_booster()
    else:
        actual_booster = model.get_booster()

    scores = actual_booster.get_score(importance_type="gain")
    feat_order = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    feat_names = [f[0] for f in feat_order]
    feat_vals  = [f[1] for f in feat_order]

    print(f"\n  Feature Importance (Gain):")
    max_gain = max(feat_vals) if feat_vals else 1
    for name, val in feat_order[:15]:
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
    plot_n = 15
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Phase 8 Analysis — {asset.upper()} (Top {plot_n} Meta-Drivers)", fontsize=14, fontweight="bold")

    # Left: Gain importance (Top 15)
    ax = axes[0]
    top_names = feat_names[:plot_n]
    top_vals  = feat_vals[:plot_n]
    colors = ["#4CAF50" for _ in range(len(top_vals))]
    ax.barh(top_names[::-1], top_vals[::-1], color=colors)
    ax.set_title("Alpha Source: Importance (Gain)", fontweight="bold")
    ax.set_xlabel("Gain")

    # Right: Correlation (Top 15)
    ax = axes[1]
    top_corr_names = [f[0] for f in sorted_corrs[:plot_n]]
    top_corr_vals  = [f[1] for f in sorted_corrs[:plot_n]]
    
    colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in top_corr_vals[::-1]]
    ax.barh(top_corr_names[::-1], top_corr_vals[::-1], color=colors)
    ax.set_title("Alpha Direction: Correlation", fontweight="bold")
    ax.set_xlabel("r")
    ax.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    out = f"{REPORT_DIR}/{asset}_meta_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Append to Log
    summary_path = f"{REPORT_DIR}/log.txt"
    with open(summary_path, "a") as sf:
        sf.write(f"\n{'='*60}\n")
        sf.write(f"ASSET: {asset.upper()}\n")
        sf.write(f"{'='*60}\n\n")
        
        sf.write("1. FEATURE IMPORTANCE (GAIN) - FULL SET:\n")
        sf.write(f"{'-'*40}\n")
        for name, val in feat_order:
            sf.write(f"  - {name:25s}: {val:10.4f}\n")
            
        sf.write("\n2. CORRELATION - FULL SET:\n")
        sf.write(f"{'-'*40}\n")
        for name, val in sorted_corrs:
            sf.write(f"  - {name:25s}: {val:+10.6f}\n")
        sf.write("\n")