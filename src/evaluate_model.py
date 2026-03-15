"""
Model Evaluation Script
========================
Loads the saved XGBoost model and generates visual plots explaining:
1. Confusion Matrix (What Accuracy, Precision, Recall ACTUALLY mean)
2. ROC Curve (Why AUC matters)
3. Probability Distribution (How confident is the model?)
4. Threshold Analysis (What happens when we change the confidence cutoff?)

All plots are saved to: reports/
"""

import polars as pl
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves to file)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "models/nifty.joblib"
TRAIN_PATH = "data/processed/train/nifty.parquet"
TEST_PATH  = "data/processed/test/nifty.parquet"
CONFIG_PATH = "config.yaml"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# ============================================
# LOAD DATA & MODEL
# ============================================
print("=" * 50)
print("LOADING DATA & MODEL")
print("=" * 50)

model = joblib.load(MODEL_PATH)
print(f"Model loaded from: {MODEL_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

features = config["Exposure_Features"]
threshold = config.get("Threshold", 0.60)
print(f"Features: {len(features)}")
print(f"Confidence Threshold: {threshold}")

# Load both datasets
train_pl = pl.read_parquet(TRAIN_PATH)
test_pl  = pl.read_parquet(TEST_PATH)

X_train = train_pl.select(features).to_pandas()
y_train = train_pl["Label"].to_pandas()
X_test  = test_pl.select(features).to_pandas()
y_test  = test_pl["Label"].to_pandas()

print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ============================================
# PREDICTIONS
# ============================================
# Probabilities (0.0 to 1.0) — How confident is the model?
train_probs = model.predict_proba(X_train)[:, 1]
test_probs  = model.predict_proba(X_test)[:, 1]

# Binary predictions at DEFAULT threshold (0.50)
train_preds_50 = (train_probs >= 0.50).astype(int)
test_preds_50  = (test_probs >= 0.50).astype(int)

# Binary predictions at CUSTOM threshold (0.60)
train_preds_custom = (train_probs >= threshold).astype(int)
test_preds_custom  = (test_probs >= threshold).astype(int)


# ============================================
# PLOT 1: CONFUSION MATRIX
# ============================================
# What is a Confusion Matrix?
# ┌────────────────────┬─────────────────┬─────────────────┐
# │                    │ Predicted DOWN  │ Predicted UP    │
# ├────────────────────┼─────────────────┼─────────────────┤
# │ Actually DOWN (0)  │ TRUE NEGATIVE   │ FALSE POSITIVE  │
# │                    │ "Correct Skip"  │ "Bad Buy"       │
# ├────────────────────┼─────────────────┼─────────────────┤
# │ Actually UP (1)    │ FALSE NEGATIVE  │ TRUE POSITIVE   │
# │                    │ "Missed Trade"  │ "Good Buy"      │
# └────────────────────┴─────────────────┴─────────────────┘

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (y_true, y_pred, title) in enumerate([
    (y_train, train_preds_50, "TRAIN SET (Threshold = 0.50)"),
    (y_test,  test_preds_50,  "TEST SET (Threshold = 0.50)")
]):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    ax = axes[idx]
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    
    # Annotate cells with counts and labels
    labels = [
        [f"TRUE NEG\n{tn}\n(Correct Skip)", f"FALSE POS\n{fp}\n(Bad Buy)"],
        [f"FALSE NEG\n{fn}\n(Missed Trade)", f"TRUE POS\n{tp}\n(Good Buy)"]
    ]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, labels[i][j], ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted DOWN', 'Predicted UP'])
    ax.set_yticklabels(['Actually DOWN', 'Actually UP'])
    ax.set_title(title, fontsize=12, fontweight='bold')

plt.suptitle("CONFUSION MATRIX\n(What the model got right and wrong)", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/1_confusion_matrix.png", dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: {REPORT_DIR}/1_confusion_matrix.png")


# ============================================
# PLOT 2: METRICS COMPARISON (Train vs Test)
# ============================================
# Accuracy  = (TP + TN) / Total         → "Overall correctness"
# Precision = TP / (TP + FP)            → "When I said BUY, was I right?"
# Recall    = TP / (TP + FN)            → "Did I catch all the UP days?"
# F1 Score  = 2 * (Prec * Rec) / (P+R)  → "Balance of Precision and Recall"

metrics = {
    'Accuracy':  [accuracy_score(y_train, train_preds_50),  accuracy_score(y_test, test_preds_50)],
    'Precision': [precision_score(y_train, train_preds_50), precision_score(y_test, test_preds_50)],
    'Recall':    [recall_score(y_train, train_preds_50),    recall_score(y_test, test_preds_50)],
    'F1 Score':  [f1_score(y_train, train_preds_50),        f1_score(y_test, test_preds_50)],
    'ROC AUC':   [roc_auc_score(y_train, train_probs),      roc_auc_score(y_test, test_probs)],
}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35

train_vals = [v[0] for v in metrics.values()]
test_vals  = [v[1] for v in metrics.values()]

bars1 = ax.bar(x - width/2, train_vals, width, label='Train', color='#4CAF50', alpha=0.85)
bars2 = ax.bar(x + width/2, test_vals,  width, label='Test',  color='#FF5722', alpha=0.85)

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

# Reference line at 0.50 (random chance)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random (0.50)')

ax.set_xticks(x)
ax.set_xticklabels(metrics.keys(), fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('MODEL METRICS: Train vs Test\n'
             '(If Train >> Test, the model is OVERFITTING)', 
             fontsize=13, fontweight='bold')
ax.set_ylim(0.40, 0.75)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/2_metrics_comparison.png", dpi=150, bbox_inches='tight')
print(f"✅ Saved: {REPORT_DIR}/2_metrics_comparison.png")


# ============================================
# PLOT 3: ROC CURVE
# ============================================
# The ROC Curve answers: "At every possible threshold, what is the tradeoff
# between catching UP days (True Positive Rate) and false alarms (False Positive Rate)?"
#
# - A PERFECT model hugs the top-left corner (AUC = 1.0)
# - A RANDOM model sits on the diagonal (AUC = 0.5)
# - Our model is somewhere in between

fig, ax = plt.subplots(figsize=(8, 7))

for y_true, probs_arr, label, color in [
    (y_train, train_probs, 'Train', '#4CAF50'),
    (y_test,  test_probs,  'Test',  '#FF5722')
]:
    fpr, tpr, thresholds = roc_curve(y_true, probs_arr)
    auc = roc_auc_score(y_true, probs_arr)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{label} (AUC = {auc:.4f})')

# Diagonal = Random guessing
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guess (AUC = 0.50)')

ax.set_xlabel('False Positive Rate\n(How often we falsely predicted UP)', fontsize=11)
ax.set_ylabel('True Positive Rate\n(How many UP days we caught)', fontsize=11)
ax.set_title('ROC CURVE\n'
             '(Further from the diagonal = Better model)\n'
             '(If Train curve >> Test curve = Overfitting)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/3_roc_curve.png", dpi=150, bbox_inches='tight')
print(f"✅ Saved: {REPORT_DIR}/3_roc_curve.png")


# ============================================
# PLOT 4: PROBABILITY DISTRIBUTION
# ============================================
# This shows HOW confident the model is on each prediction.
# - A GOOD model: Two separate peaks (one near 0, one near 1). It is decisive.
# - A BAD model: One big peak near 0.50. It is always unsure.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (probs_arr, y_true, title) in enumerate([
    (train_probs, y_train, "TRAIN SET"),
    (test_probs,  y_test,  "TEST SET")
]):
    ax = axes[idx]
    
    # Split by actual class
    up_probs   = probs_arr[y_true == 1]
    down_probs = probs_arr[y_true == 0]
    
    ax.hist(down_probs, bins=40, alpha=0.6, color='red',  label='Actually DOWN', density=True)
    ax.hist(up_probs,   bins=40, alpha=0.6, color='green', label='Actually UP',   density=True)
    
    # Threshold lines
    ax.axvline(0.50, color='gray',  linestyle='--', linewidth=1.5, label='Default (0.50)')
    ax.axvline(threshold, color='blue', linestyle='--', linewidth=1.5, label=f'Our Threshold ({threshold})')
    
    ax.set_xlabel('Model Probability (of UP)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{title}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle("PROBABILITY DISTRIBUTION\n"
             "(Good model = Green peak at RIGHT, Red peak at LEFT)\n"
             "(Bad model = Both peaks overlap near 0.50)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/4_probability_distribution.png", dpi=150, bbox_inches='tight')
print(f"✅ Saved: {REPORT_DIR}/4_probability_distribution.png")


# ============================================
# PLOT 5: THRESHOLD SWEEP
# ============================================
# What happens if we change the confidence threshold?
# Lower threshold = More trades, lower win rate
# Higher threshold = Fewer trades, higher win rate (hopefully)

thresholds_to_test = np.arange(0.45, 0.70, 0.01)
results = []

for t in thresholds_to_test:
    mask = test_probs >= t
    n = mask.sum()
    if n > 10:  # Need at least 10 trades to be meaningful
        win_rate = y_test[mask].mean()  # What % of filtered days were UP
        results.append({
            'threshold': t,
            'n_trades': n,
            'win_rate': win_rate
        })

df_results = pd.DataFrame(results)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Win Rate (left axis)
color1 = '#FF5722'
ax1.plot(df_results['threshold'], df_results['win_rate'] * 100, 
         color=color1, linewidth=2.5, marker='o', markersize=4, label='Win Rate (%)')
ax1.set_xlabel('Confidence Threshold', fontsize=12)
ax1.set_ylabel('Win Rate (%)', color=color1, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')

# Number of Trades (right axis)
ax2 = ax1.twinx()
color2 = '#2196F3'
ax2.bar(df_results['threshold'], df_results['n_trades'], 
        width=0.008, alpha=0.3, color=color2, label='# Trades')
ax2.set_ylabel('Number of Trades', color=color2, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color2)

# Mark our chosen threshold
ax1.axvline(x=threshold, color='green', linestyle='-', linewidth=2, alpha=0.7)
ax1.text(threshold + 0.005, ax1.get_ylim()[1] - 2, 
         f'← Our Threshold ({threshold})', color='green', fontsize=10, fontweight='bold')

ax1.set_title('THRESHOLD SWEEP\n'
              '(Higher threshold = Fewer but better trades)\n'
              '(Find the sweet spot: good win rate + enough trades)',
              fontsize=13, fontweight='bold')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/5_threshold_sweep.png", dpi=150, bbox_inches='tight')
print(f"✅ Saved: {REPORT_DIR}/5_threshold_sweep.png")


# ============================================
# SUMMARY REPORT (Text)
# ============================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

baseline = y_test.mean()

print(f"\nDataset:")
print(f"  Train Size: {len(y_train)} rows")
print(f"  Test Size:  {len(y_test)} rows")
print(f"  Baseline (% UP days in test): {baseline*100:.1f}%")

print(f"\nDefault Threshold (0.50):")
print(f"  Accuracy:  {accuracy_score(y_test, test_preds_50):.4f}")
print(f"  Precision: {precision_score(y_test, test_preds_50):.4f}")
print(f"  Recall:    {recall_score(y_test, test_preds_50):.4f}")
print(f"  ROC AUC:   {roc_auc_score(y_test, test_probs):.4f}")

# High confidence stats
mask = test_probs >= threshold
if mask.sum() > 0:
    hc_win = y_test[mask].mean()
    print(f"\nHigh Confidence (>{threshold}):")
    print(f"  Trades:    {mask.sum()} / {len(y_test)}")
    print(f"  Win Rate:  {hc_win*100:.1f}%")
    print(f"  vs Baseline: {'+' if hc_win > baseline else ''}{(hc_win - baseline)*100:.1f}%")

print(f"\nOverfitting Check:")
train_acc = accuracy_score(y_train, train_preds_50)
test_acc  = accuracy_score(y_test, test_preds_50)
gap = train_acc - test_acc
print(f"  Train Acc: {train_acc:.4f}")
print(f"  Test Acc:  {test_acc:.4f}")
print(f"  Gap:       {gap:.4f}", end="")
if gap > 0.10:
    print(" ⚠️  OVERFITTING (Train >> Test)")
elif gap > 0.05:
    print(" ⚠️  Mild overfitting")
else:
    print(" ✅ Healthy (Train ≈ Test)")

print(f"\nVerdict:")
if roc_auc_score(y_test, test_probs) > 0.55:
    print("  ✅ Model has a meaningful edge")
elif roc_auc_score(y_test, test_probs) > 0.52:
    print("  ⚠️  Model has a weak but real edge")
else:
    print("  ❌ Model is near random — consider better features or target")

print(f"\n📊 All plots saved to: {REPORT_DIR}/")
print("=" * 60)
