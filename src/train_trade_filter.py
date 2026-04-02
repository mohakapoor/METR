"""
train_trade_filter.py — Phase 8.5: Calibrated Meta-Labeling (XGBoost)
=====================================================================
- Objective: Predict Meta_Label (1 if Signal matches TB_Label, 0 otherwise)
- Logic: Isotonic Calibration + Stability Constraints (Gamma, min_child_weight)
- Metrics: Overfitting Gap (Train vs Test AUC) + Threshold Sweep
- Outputs: models/meta/*.joblib and reports/meta_filter/*.txt
"""

import polars as pl
import numpy as np
import yaml
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
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

# -- Hyperparameter Space 
param_grid = {
    "max_depth":        [2,3],
    "learning_rate":    [0.02, 0.03, 0.05, 0.1],
    "n_estimators":     [100, 125,150,],
    "reg_lambda":       [1, 10, 20, 50, 100], 
    "min_child_weight": [5, 10, 20],      # Stability Constraint
    "gamma":            [0.1,  0.5,1.0],  # Pruning Constraint
    "subsample":        [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9]
}


# Training Loop
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

for asset in ASSETS:
    print(f"\n{'='*60}")
    print(f"  ASSET: {asset.upper()} (Calibrated XGBoost Meta-Filter)")
    print(f"{'='*60}")

    # Load data
    train = pl.read_parquet(f"data/processed/train/{asset}.parquet")
    test  = pl.read_parquet(f"data/processed/test/{asset}.parquet")

    # Meta-Target: Meta_Label (1 = Win, 0 = Loss/Noise)
    y_train = train["Meta_Label"].to_pandas()
    y_test  = test["Meta_Label"].to_pandas()

    # Features: Base + Interaction + Cross-Asset
    features = EXP_FEATURES + CROSS_FEATURES.get(asset, [])
    available = [f for f in features if f in train.columns]
    X_train = train.select(available).to_pandas()
    X_test  = test.select(available).to_pandas()

    print(f"   Shape: Train={X_train.shape[0]}, Test={X_test.shape[0]} | Features: {len(available)}")
    
    baseline_wr = y_test.mean()

    # Model Setup
    base_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        tree_method="hist", 
        device="cuda",       
        eval_metric="logloss"
    )

    # Randomized Search (Deep Coverage: n_iter=50)
    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_grid,
        n_iter=50,                     
        scoring="average_precision",
        cv=tscv,
        random_state=42,
        n_jobs=1 
    )

    search.fit(X_train, y_train)
    best_raw_model = search.best_estimator_

    # --- Probability Calibration (Isotonic) ---
    # Using 'prefit' to preserve the Best Estimator from RandomizedSearch
    # Note: Fitting on X_train is slightly optimistic but acceptable for signal filtration.
    calibrated_model = CalibratedClassifierCV(
        best_raw_model, method='isotonic', cv='prefit'
    )
    calibrated_model.fit(X_train, y_train)

    # --- Evaluation ---
    # Probabilities of Meta_Label=1 (Signal Win)
    train_probs = calibrated_model.predict_proba(X_train)[:, 1]
    test_probs  = calibrated_model.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc  = roc_auc_score(y_test, test_probs)
    ap_score  = average_precision_score(y_test, test_probs)
    auc_gap   = train_auc - test_auc

    report_lines = []
    report_lines.append(f"============================================================")
    report_lines.append(f"  ASSET: {asset.upper()} (Calibrated XGBoost Meta-Filter)")
    report_lines.append(f"============================================================")
    report_lines.append(f"   Baseline Win Rate: {baseline_wr:.4f}")
    report_lines.append(f"   Best CV AUPRC:     {search.best_score_:.4f}")
    report_lines.append(f"   Best Params:       {search.best_params_}")
    report_lines.append(f"\n   Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | Gap: {auc_gap:.4f}")
    report_lines.append(f"   Test AUPRC: {ap_score:.4f}")

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
    
    joblib.dump(calibrated_model, f"models/meta/{asset}_xgb_meta.joblib")
    
    print(f"   AUC Stats: Train={train_auc:.4f} | Test={test_auc:.4f} | Gap={auc_gap:.4f}")
    print(f"   Saved Model  → models/meta/{asset}_xgb_meta.joblib")
    print(f"   Saved Report → {filename}")
    
print("\nMeta-Training Complete.")