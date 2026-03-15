"""
Feature Importance Analysis
Extracts what the model learned and shows which features matter.
"""
import polars as pl
import pandas as pd
import numpy as np
import joblib
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load
model = joblib.load("models/nifty2.joblib")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
features = config["Exposure_Features"]

# --- 1. Feature Importance (3 types) ---
# "gain" = how much each feature improves accuracy when used in a split
# "weight" = how many times each feature is used across all trees  
# "cover" = how many data points each feature affects

for imp_type in ['gain', 'weight', 'cover']:
    scores = model.get_booster().get_score(importance_type=imp_type)
    # Model already stores real feature names (not f0, f1...)
    named_scores = scores
    
    # Sort
    sorted_feats = sorted(named_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n=== Feature Importance ({imp_type.upper()}) ===")
    for name, score in sorted_feats:
        bar = "█" * int(score / max(named_scores.values()) * 30)
        print(f"  {name:20s} {score:10.2f}  {bar}")

# --- 2. Correlation with Label ---
test = pl.read_parquet("data/processed/test/nifty.parquet")
train = pl.read_parquet("data/processed/train/nifty.parquet")

print(f"\n=== Feature Correlation with Label (Test Set) ===")
correlations = {}
for feat in features:
    corr = test.select(pl.corr(feat, "Label")).item()
    correlations[feat] = corr

sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
for name, corr in sorted_corr:
    direction = "↑ UP" if corr > 0 else "↓ DOWN"
    bar = "█" * int(abs(corr) * 100)
    print(f"  {name:20s} {corr:+.4f}  {direction:6s}  {bar}")

# --- 3. Plot ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gain importance
scores = model.get_booster().get_score(importance_type='gain')
named_scores = scores
sorted_feats = sorted(named_scores.items(), key=lambda x: x[1])

ax = axes[0]
ax.barh([f[0] for f in sorted_feats], [f[1] for f in sorted_feats], color='#4CAF50')
ax.set_title("Feature Importance (GAIN)\n(How much each feature improves predictions)", fontweight='bold')
ax.set_xlabel("Gain")

# Correlation
ax = axes[1]
sorted_corr_plot = sorted(correlations.items(), key=lambda x: x[1])
colors = ['#FF5722' if c < 0 else '#4CAF50' for _, c in sorted_corr_plot]
ax.barh([f[0] for f in sorted_corr_plot], [f[1] for f in sorted_corr_plot], color=colors)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title("Feature Correlation with Label\n(Green = helps predict UP, Red = helps predict DOWN)", fontweight='bold')
ax.set_xlabel("Correlation")

plt.tight_layout()
plt.savefig("reports/6_feature_analysis.png", dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: reports/6_feature_analysis.png")
