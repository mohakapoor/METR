"""
train_baseline.py 
=================================================================
- Standard: Signal-Conditional (+1/-1/0) & Symmetric Alpha.
- Metrics: AUC and AUPRC (Average Precision).
"""

import polars as pl
import numpy as np
import yaml
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score

CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
EXP_FEATURES   = config.get("Exposure_Features", [])
CROSS_FEATURES = config.get("Cross_Asset_Features", {})
N_SPLITS       = config.get("N_Splits", 5)
ASSETS = ["nifty", "gold", "usdinr"]

global_log = []

# Baseline Search Grid
param_grid = {
    "logreg__C": [0.01, 0.1, 1.0, 10.0],
    "logreg__solver": ["lbfgs"]
}

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

for asset in ASSETS:
    print(f"\n{'='*60}")
    print(f"  ASSET: {asset.upper()} (Symmetric Baseline)")
    print(f"{'='*60}")
    
    global_log.append(f"\n{'='*60}")
    global_log.append(f"  ASSET: {asset.upper()} (Symmetric Baseline)")
    global_log.append(f"{'='*60}")


    train = pl.read_parquet(f"data/processed/train/{asset}.parquet")
    test  = pl.read_parquet(f"data/processed/test/{asset}.parquet")

    # [SIGNAL MASKING]
    train = train.filter(pl.col("Signal") != 0)
    test  = test.filter(pl.col("Signal") != 0)

    y_train = train["Meta_Label"].to_pandas()
    y_test  = test["Meta_Label"].to_pandas()
    
    features = EXP_FEATURES + CROSS_FEATURES.get(asset, [])
    available = [f for f in features if f in train.columns]
    X_train  = train.select(available).to_pandas()
    X_test   = test.select(available).to_pandas()

    # Model Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, random_state=42))
    ])

    search = GridSearchCV(
        estimator=pipe, param_grid=param_grid,
        scoring="average_precision", cv=tscv, n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_raw_model = search.best_estimator_

    # Out-Of-Fold (OOF) Probability Generation
    print(f"   Generating Calibrated OOF probabilities for {asset}...")
    oof_probs = np.full(X_train.shape[0], np.nan)
    
    for train_idx, test_idx in tscv.split(X_train, y_train):
        fold_calib = CalibratedClassifierCV(best_raw_model, method='isotonic', cv=3)
        fold_calib.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof_probs[test_idx] = fold_calib.predict_proba(X_train.iloc[test_idx])[:, 1]
    
    # SAVE DIRECTIONAL BLOBS
    val_blob = pl.DataFrame({"OOF_Prob": oof_probs, "Directional_Return": train["Directional_Return"]})
    val_blob = val_blob.filter(pl.col("OOF_Prob").is_not_nan())
    val_blob.write_parquet(f"models/baseline/{asset}_val_blob.parquet")

    # Final Production Model
    calibrated_model = CalibratedClassifierCV(
        best_raw_model, 
        method='isotonic', 
        cv=3
    )
    calibrated_model.fit(X_train, y_train)
    
    # Evaluation
    train_probs = calibrated_model.predict_proba(X_train)[:, 1]
    test_probs  = calibrated_model.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc  = roc_auc_score(y_test, test_probs)
    train_ap  = average_precision_score(y_train, train_probs)
    test_ap   = average_precision_score(y_test, test_probs)
    
    auc_trace = f"   AUC Trace: Train={train_auc:.4f} | Test={test_auc:.4f} | Gap={train_auc - test_auc:.4f}"
    ap_trace  = f"   AP  Trace: Train={train_ap:.4f} | Test={test_ap:.4f} | Gap={train_ap - test_ap:.4f}"
    
    print(f"\n{auc_trace}")
    print(ap_trace)
    
    global_log.append(auc_trace)
    global_log.append(ap_trace)
    global_log.append(f"   [DONE] Artifacts saved in models/baseline/")

    joblib.dump(calibrated_model, f"models/baseline/{asset}_logreg_baseline.joblib")

# Save Cumulative Results Log
log_path = "reports/baseline/results.txt"
with open(log_path, "w") as f:
    f.write("\n".join(global_log))
