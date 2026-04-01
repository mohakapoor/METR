"""
feature_analysis.py — Multi-Asset Feature Analysis
====================================================
Loads each asset's unified model + test data and generates:
  1. Feature Importance (Gain)
  2. Feature vs Label Correlation (one-vs-rest: -1, 0, +1)
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

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

EXP_FEATURES   = config["Exposure_Features"]
CROSS_FEATURES = config["Cross_Asset_Features"]

ASSETS = ["nifty", "gold", "usdinr"]
REPORT_DIR = "reports/feature_analysis"
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Per-Asset Analysis ────────────────────────────────────────────────────────
for asset in ASSETS:
    model_path = f"models/{asset}_unified.joblib"
    if not os.path.exists(model_path):
        print(f"[SKIP] {asset} — model not found at {model_path}")
        continue

    model = joblib.load(model_path)
    test  = pl.read_parquet(f"data/processed/test/{asset}.parquet")

    features  = EXP_FEATURES + CROSS_FEATURES.get(asset, [])
    available = list(dict.fromkeys(f for f in features if f in test.columns))  # dedup + order

    # Only rows with valid labels
    test = test.filter(pl.col("Label").is_not_null())
    labels = test["Label"].to_numpy()           # original: -1, 0, +1
    X = test.select(available).to_pandas()

    print(f"\n{'='*60}")
    print(f"  {asset.upper()} | {len(available)} features | {len(labels)} test rows")
    print(f"{'='*60}")

    # ── 1. Feature Importance (Gain) ──────────────────────────────────────────
    scores = model.get_booster().get_score(importance_type="gain")
    # Align to available features (model may have subset)
    feat_order = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    feat_names = [f[0] for f in feat_order]
    feat_vals  = [f[1] for f in feat_order]

    print(f"\n  Feature Importance (Gain):")
    max_gain = max(feat_vals) if feat_vals else 1
    for name, val in feat_order:
        bar = "█" * int(val / max_gain * 30)
        print(f"    {name:25s} {val:8.2f}  {bar}")

    # ── 2. One-vs-Rest Correlations with Label ────────────────────────────────
    # Binary encode each class vs rest
    corr_data = {}
    for cls, cls_name in [(-1, "vs -1(SL)"), (0, "vs 0(TO)"), (1, "vs +1(TP)")]:
        binary = (labels == cls).astype(float)
        corrs = {}
        for feat in available:
            col = X[feat].values.astype(float)
            mask = ~np.isnan(col)
            if mask.sum() > 10:
                corrs[feat] = np.corrcoef(col[mask], binary[mask])[0, 1]
            else:
                corrs[feat] = 0.0
        corr_data[cls_name] = corrs

    print(f"\n  Feature Correlations (one-vs-rest):")
    print(f"    {'Feature':25s} {'vs -1':>8} {'vs 0':>8} {'vs +1':>8}")
    print(f"    {'-'*55}")
    for feat in available:
        r_sl = corr_data["vs -1(SL)"].get(feat, 0)
        r_to = corr_data["vs 0(TO)"].get(feat, 0)
        r_tp = corr_data["vs +1(TP)"].get(feat, 0)
        print(f"    {feat:25s} {r_sl:+8.4f} {r_to:+8.4f} {r_tp:+8.4f}")

    # ── 3. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(available) * 0.35)))
    fig.suptitle(f"Feature Analysis — {asset.upper()} (Unified Model)", fontsize=14, fontweight="bold")

    # Left: Gain importance (horizontal bar)
    ax = axes[0]
    colors = ["#4CAF50" if v > np.median(feat_vals) else "#90CAF9" for v in feat_vals[::-1]]
    ax.barh(feat_names[::-1], feat_vals[::-1], color=colors)
    ax.set_title("Feature Importance (Gain)\nGreen = above median", fontweight="bold")
    ax.set_xlabel("Gain")
    ax.axvline(np.median(feat_vals), color="gray", linestyle="--", alpha=0.6, label="Median")
    ax.legend(fontsize=8)

    # Right: Correlation heatmap-style grouped bar
    ax = axes[1]
    feat_list = available
    n = len(feat_list)
    x = np.arange(n)
    w = 0.28

    r_sl = [corr_data["vs -1(SL)"].get(f, 0) for f in feat_list]
    r_to = [corr_data["vs 0(TO)"].get(f, 0)  for f in feat_list]
    r_tp = [corr_data["vs +1(TP)"].get(f, 0) for f in feat_list]

    ax.barh(x - w, r_sl, w, label="-1 (Stop-Loss)", color="#e74c3c", alpha=0.8)
    ax.barh(x,     r_to, w, label=" 0 (Timeout)",   color="#95a5a6", alpha=0.8)
    ax.barh(x + w, r_tp, w, label="+1 (Profit)",    color="#2ecc71", alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(feat_list, fontsize=8)
    ax.set_title("Feature Correlation (One-vs-Rest)\nGreen bars right = predicts profit", fontweight="bold")
    ax.set_xlabel("Correlation coefficient")
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out = f"{REPORT_DIR}/{asset}_feature_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ Saved → {out}")
