import polars as pl
import numpy as np
import yaml
import os
from sklearn.metrics import roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    print(f"Error: {CONFIG_PATH} not found.")
    exit(1)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

EXP_FEATURES   = config.get("Exposure_Features", [])
CROSS_FEATURES = config.get("Cross_Asset_Features", {})
ASSETS = ["nifty", "gold", "usdinr"]

print("============================================================")
print("  FEATURE SEPARATION ANALYSIS: META-LABEL (WIN VS LOSS)")
print("  Metric: ROC AUC (|AUC - 0.5|)")
print("============================================================\n")

for asset in ASSETS:
    train_path = f"data/processed/train/{asset}.parquet"
    if not os.path.exists(train_path):
        print(f"[SKIP] {asset.upper()} — Train data not found at {train_path}")
        continue

    df = pl.read_parquet(train_path)
    
    # Meta_Label: 1 (Winning Signal), 0 (Losing/Timeout Signal)
    # We drop nulls but keep all 0s and 1s to see the full "Filter" power
    df_meta = df.filter(pl.col("Meta_Label").is_not_null())
    
    if df_meta.height < 10:
        print(f"[SKIP] {asset.upper()} — Not enough meta-labeled samples ({df_meta.height})")
        continue

    y = df_meta["Meta_Label"].to_numpy().astype(int)
    features = EXP_FEATURES + CROSS_FEATURES.get(asset, [])
    # Deduplicate while preserving order
    features = list(dict.fromkeys(f for f in features if f in df.columns))

    results = []
    for f in features:
        vals = df_meta[f].to_numpy()
        
        # Handle NaNs/Infs
        mask = np.isfinite(vals)
        if mask.sum() < 10:
            continue
            
        try:
            auc = roc_auc_score(y[mask], vals[mask])
            separation = abs(auc - 0.5)
            direction = "Higher = Win Conviction" if auc > 0.5 else "Higher = Fail/Noise"
            results.append({
                "feature": f,
                "auc": auc,
                "separation": separation,
                "direction": direction
            })
        except ValueError:
            continue

    # Sort by separation
    top_5 = sorted(results, key=lambda x: x["separation"], reverse=True)[:5]

    print(f"ASSET: {asset.upper()}")
    print(f"{'-'*60}")
    if not top_5:
        print("  No features with measurable meta-separation found.")
    else:
        print(f"{'Feature':25s} | {'AUC':>6} | {'Separation':>10} | {'Direction'}")
        print(f"{'-'*75}")
        for r in top_5:
            print(f"{r['feature']:25s} | {r['auc']:6.3f} | {r['separation']:10.3f} | {r['direction']}")
    print("\n")

print("Analysis Complete.")
