"""
Trade Filtering Model
=====================
- Only execute when Signal is there 
- Calc AUCROC and AUC
"""

import polars as pl
import numpy as np
import yaml
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import xgboost as xgb

# Config
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

EXP_FEATURES   = config.get("Exposure_Features", [])
CROSS_FEATURES = config.get("Cross_Asset_Features", {})
N_SPLITS = config.get("N_Splits", 5)
ASSETS = ["nifty", "gold", "usdinr"]


# -- Hyperparameter Space 
param_grid = {
    "max_depth":        [2,3],
    "learning_rate":    [0.02, 0.03, 0.05, 0.1],
    "n_estimators":     [100, 125,150,],
    "reg_lambda":       [1, 10, 20, 50, 100],
    "min_child_weight": [5, 10, 20],
    "gamma":            [0.1,  0.5,1.0],
    "subsample":        [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
}


# Training Loop
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

for asset in ASSETS:
    print(f"\n{'='*50}")
    print(f"  ASSET: {asset.upper()}")
    print(f"{'='*50}")

    # Load data
    train = pl.read_parquet(f"data/processed/train/{asset}.parquet")
    test  = pl.read_parquet(f"data/processed/test/{asset}.parquet")

    # SIGNAL MASKING
    train = train.filter(pl.col("Signal") != 0)
    test  = test.filter(pl.col("Signal") != 0)

    # Meta-Target: Meta_Label (1 = Win, 0 = Loss/Noise)
    y_train = train["Meta_Label"].to_pandas()
    y_test  = test["Meta_Label"].to_pandas()

    # Features: Base + Interaction + Cross-Asset
    features = EXP_FEATURES + CROSS_FEATURES.get(asset, [])
    available = [f for f in features if f in train.columns]
    X_train = train.select(available).to_pandas()
    X_test  = test.select(available).to_pandas()

    print(f"   Signals Found: Train={X_train.shape[0]}, Test={X_test.shape[0]} | Features: {len(available)}")
    
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

    # Out-Of-Fold  Probability Generation    
    oof_probs = np.full(X_train.shape[0], np.nan)
    
    # TimeSeriesSplit
    for train_idx, test_idx in tscv.split(X_train, y_train):
        fold_calib = CalibratedClassifierCV(best_raw_model, method='isotonic', cv=3)
        fold_calib.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        
        # Predict on the "Future" fold
        oof_probs[test_idx] = fold_calib.predict_proba(X_train.iloc[test_idx])[:, 1]
    
    # SAVE VALIDATION BLOBS FOR SHARPE OPTIMIZER
    val_blob = pl.DataFrame({
        "OOF_Prob": oof_probs,
        "Directional_Return": train["Directional_Return"],
        "Date": train["Date"],
    })
    # Filter out NaN points 
    val_blob = val_blob.filter(pl.col("OOF_Prob").is_not_nan())
    val_blob.write_parquet(f"models/meta/{asset}_val_blob.parquet")
    print(f"   Saved OOF Blob → models/meta/{asset}_val_blob.parquet ({val_blob.shape[0]} signals)")

    # Probability Calibration (Isotonic)
    calibrated_model = CalibratedClassifierCV(
        best_raw_model, method='isotonic', cv='prefit'
    )
    calibrated_model.fit(X_train, y_train)

    # Evaluation
    train_probs = calibrated_model.predict_proba(X_train)[:, 1]
    test_probs  = calibrated_model.predict_proba(X_test)[:, 1]
    
    # SAVE TEST BLOBS
    test_blob = pl.DataFrame({
        "OOF_Prob": test_probs,
        "Directional_Return": test["Directional_Return"],
        "Date": test["Date"]
    })
    test_blob.write_parquet(f"models/meta/{asset}_test_blob.parquet")
    print(f"   Saved Test Blob → models/meta/{asset}_test_blob.parquet ({test_blob.shape[0]} signals)")

    train_auc = roc_auc_score(y_train, train_probs)
    test_auc  = roc_auc_score(y_test, test_probs)
    train_ap  = average_precision_score(y_train, train_probs)
    test_ap   = average_precision_score(y_test, test_probs)
    auc_gap   = train_auc - test_auc
    ap_gap    = train_ap - test_ap

    report_lines = []
    report_lines.append(f"======================================")
    report_lines.append(f"  ASSET: {asset.upper()} Train Filter ")
    report_lines.append(f"======================================")
    report_lines.append(f"   Baseline Win Rate: {baseline_wr:.4f}")
    report_lines.append(f"   Best CV AUPRC:     {search.best_score_:.4f}")
    report_lines.append(f"   Best Params:       {search.best_params_}")
    report_lines.append(f"\n   Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | Gap: {auc_gap:.4f}")
    report_lines.append(f"   Train AP:  {train_ap:.4f} | Test AP:  {test_ap:.4f} | Gap: {ap_gap:.4f}")

    # Save Report
    filename = f"reports/meta_filter/{asset}_xgb_report.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("\n".join(report_lines))
    
    joblib.dump(calibrated_model, f"models/meta/{asset}_xgb_meta.joblib")
    
    print(f"   AUC Trace: Train={train_auc:.4f} | Test={test_auc:.4f} | Gap={auc_gap:.4f}")
    print(f"   AP  Trace: Train={train_ap:.4f} | Test={test_ap:.4f} | Gap={ap_gap:.4f}")
    print(f"   Saved Model  → models/meta/{asset}_xgb_meta.joblib")
    print(f"   Saved Report → {filename}")
    
print("\nMeta-Training Complete.")