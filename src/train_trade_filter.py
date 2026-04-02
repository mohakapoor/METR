"""
train_trade_filter.py — Phase 7: Binary Meta-Labeling (XGBoost)
==============================================================
- Objective: Predict Meta_Label (1 if Signal matches TB_Label, 0 otherwise)
- Logic: Insight-driven hyperparameters (Exhaustion detection, lower LR)
- Metrics: Threshold Sweep (0.50, 0.52, 0.55, 0.58)
- Outputs: models/*.joblib and reports/meta_filter/*.txt
"""

import polars as pl
import numpy as np
import yaml
import joblib
import os
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import recall_score, precision_score, roc_auc_score, average_precision_score, classification_report
import xgboost as xgb

# Config
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

EXP_FEATURES   = config.get("Exposure_Features", [])
CROSS_FEATURES = config.get("Cross_Asset_Features", {})
N_SPLITS       = config.get("N_Splits", 5)
THRESHOLD_LIST = [0.50, 0.52, 0.55, 0.58]
ASSETS = ["nifty", "gold", "usdinr"]

# Ensure output directories exist
os.makedirs("models/meta", exist_ok=True)
os.makedirs("reports/meta_filter", exist_ok=True)

# Hyperparameter Space
param_grid = {
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.02, 0.03, 0.05, 0.1],
    "n_estimators":     [100, 150, 200, 300, 500],
    "reg_lambda":       [1, 5, 10, 20, 50], 
    "subsample":        [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9]
}

# Training Loop
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

for asset in ASSETS:
    print(f"\n{'='*60}")
    print(f"  ASSET: {asset.upper()} (XGBoost Meta-Filter)")
    print(f"{'='*60}")

    # Load data
    train = pl.read_parquet(f"data/processed/train/{asset}.parquet")
    test  = pl.read_parquet(f"data/processed/test/{asset}.parquet")

    # Meta-Target: Meta_Label (1 = Win, 0 = Loss/Noise)
    y_train = train["Meta_Label"].to_pandas()
    y_test  = test["Meta_Label"].to_pandas()

    # Features: Base + Interaction + Cross-Asset
    features = EXP_FEATURES + CROSS_FEATURES.get(asset, [])
    
    # Filter to available columns
    available = [f for f in features if f in train.columns]
    X_train = train.select(available).to_pandas()
    X_test  = test.select(available).to_pandas()

    print(f"   Shape: Train={X_train.shape[0]}, Test={X_test.shape[0]} | Features: {len(available)}")
    
    # Baseline Win Rate in Test Set
    baseline_wr = y_test.mean()

    # Model Setup
    base_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        tree_method="hist", # Use binned histograms 
        device="cuda",       
        eval_metric="logloss"
    )

    # Randomized Search (Faster & Better Coverage)
    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_grid, #
        n_iter=20,                     
        scoring="average_precision",
        cv=tscv,
        random_state=42,
        n_jobs=1 
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Evaluation
    # Probabilities of Meta_Label=1 (Signal Win)
    test_probs = best_model.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, test_probs)
    ap_score  = average_precision_score(y_test, test_probs)

    report_lines = []
    report_lines.append(f"============================================================")
    report_lines.append(f"  ASSET: {asset.upper()} (XGBoost Meta-Filter)")
    report_lines.append(f"============================================================")
    report_lines.append(f"   Baseline Win Rate: {baseline_wr:.4f}")
    report_lines.append(f"   Best CV AUPRC:     {search.best_score_:.4f}")
    report_lines.append(f"   Best Params:       {search.best_params_}")
    report_lines.append(f"\n   Test AUC: {auc_score:.4f} | Test AUPRC: {ap_score:.4f}")

    for thresh in THRESHOLD_LIST:
        mask = test_probs >= thresh
        trade_count = mask.sum()
        
        if trade_count > 0:
            filtered_wr = y_test[mask].mean()
            trade_frac  = trade_count / len(y_test)
            diff = (filtered_wr - baseline_wr) * 100
            
            report_lines.append(f"\n   --- Metrics at Threshold: {thresh} ---")
            report_lines.append(f"   Win Rate: {filtered_wr:.4f} | Trade Fraction: {trade_frac:.4f}")
            report_lines.append(f"   Edge vs Baseline: {diff:+.2f}% ({trade_count} trades)")
        else:
            report_lines.append(f"\n   --- Metrics at Threshold: {thresh} ---")
            report_lines.append(f"   Zero recall (No trades selected)")

    # Save
    filename = f"reports/meta_filter/{asset}_xgb_report.txt"
    with open(filename, "w") as f:
        f.write("\n".join(report_lines))
    
    joblib.dump(best_model, f"models/meta/{asset}_xgb_meta.joblib")
    
    print(f"   AUC Stats: AUC={auc_score:.4f} | AUPRC={ap_score:.4f}")
    print(f"   ✅ Saved Model  → models/meta/{asset}_xgb_meta.joblib")
    print(f"   ✅ Saved Report → {filename}")
    
print("\nMeta-Training Complete.")