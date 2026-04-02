"""
train_baseline.py — Logistic Regression Baseline for Meta-Labeling
=================================================================
- Goal: provide a linear baseline to compare against the XGBoost Meta-Filter.
- Preprocessing: StandardScaler (Required for LogReg)
- Target: Meta_Label (1 = Winning Signal, 0 = Losing/Noise Signal)
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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    print(f"Error: {CONFIG_PATH} not found.")
    exit(1)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

EXP_FEATURES   = config["Exposure_Features"]
CROSS_FEATURES = config["Cross_Asset_Features"]
N_SPLITS       = config.get("N_Splits", 5)
THRESHOLD      = config.get("Threshold", 0.6)

ASSETS = ["nifty", "gold", "usdinr"]

# ── Hyperparameter Grid for Logistic Regression ──────────────────────────────
# 'C' is inverse regularization (smaller = stronger regularization)
param_grid = {
    "logreg__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "logreg__penalty": ["l2"], # standard ridge regularization
    "logreg__solver": ["lbfgs"]
}

# ── Training Loop ─────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

for asset in ASSETS:
    print(f"\n{'='*60}")
    print(f"  ASSET: {asset.upper()} (Logistic Regression Baseline)")
    print(f"{'='*60}")

    train_path = f"data/processed/train/{asset}.parquet"
    test_path  = f"data/processed/test/{asset}.parquet"

    if not os.path.exists(train_path):
        print(f"[SKIP] {asset.upper()} — Train data not found at {train_path}")
        continue

    train = pl.read_parquet(train_path)
    test  = pl.read_parquet(test_path)

    # Use Meta_Label as the target (Binary 0/1)
    y_train = train["Meta_Label"].to_numpy()
    y_test  = test["Meta_Label"].to_numpy()

    # Drop nulls (last T rows)
    valid_train = ~np.isnan(y_train)
    valid_test  = ~np.isnan(y_test)
    y_train = y_train[valid_train].astype(int)
    y_test  = y_test[valid_test].astype(int)

    # Use the features defined for this asset
    features = EXP_FEATURES + CROSS_FEATURES.get(asset, [])
    available = [f for f in features if f in train.columns]

    X_train = train.select(available).to_pandas().iloc[valid_train]
    X_test  = test.select(available).to_pandas().iloc[valid_test]

    print(f"   Shape: Train={X_train.shape[0]}, Test={X_test.shape[0]} | Target: Meta_Label")
    print(f"   Baseline Win Rate: {y_train.mean():.4f}")

    # Build Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42))
    ])

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=tscv,
        verbose=0,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"\n   Best CV ROC AUC: {search.best_score_:.4f}")
    print(f"   Best Params: {search.best_params_}")

    # ── Evaluation & Reporting ────────────────────────────────────────────────
    test_probs  = best_model.predict_proba(X_test)[:, 1]
    train_probs = best_model.predict_proba(X_train)[:, 1]

    train_auc = roc_auc_score(y_train, train_probs)
    test_auc  = roc_auc_score(y_test,  test_probs)

    # Ensure report directory exists
    report_dir = "reports/baseline"
    os.makedirs(report_dir, exist_ok=True)
    report_path = f"{report_dir}/{asset}_baseline_report.txt"

    with open(report_path, "w") as f_out:
        header = f"ASSET: {asset.upper()} (Logistic Regression Baseline)\n" + "="*60 + "\n"
        f_out.write(header)
        f_out.write(f"Best CV ROC AUC: {search.best_score_:.4f}\n")
        f_out.write(f"Best Params: {search.best_params_}\n")
        f_out.write(f"Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}\n\n")

        print(f"\n   AUC Stats:")
        print(f"   Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}")

        for t in [0.50, 0.52, 0.55, 0.58]:
            t_preds = (test_probs >= t).astype(int)
            
            # User Metrics: Win Rate & Trade Fraction
            # Win Rate is Precision of the predicted '1's
            # Trade Fraction is the percentage of total rows where we take the trade
            trades_taken = t_preds == 1
            if trades_taken.any():
                win_rate = y_test[trades_taken].mean()
            else:
                win_rate = 0.0
            
            trade_fraction = trades_taken.mean()

            t_report = (
                f"--- Metrics at Threshold: {t} ---\n"
                f"Win Rate: {win_rate:.4f} | Trade Fraction: {trade_fraction:.4f}\n"
            )
            class_rep = classification_report(y_test, t_preds, target_names=["Fail (0)", "Win (1)"], zero_division=0)
            
            f_out.write(t_report + class_rep + "\n")
            print(f"\n   {t_report}{class_rep}")

    # ── Save Model ────────────────────────────────────────────────────────────
    save_path = f"models/baseline/{asset}_baseline.joblib"
    joblib.dump(best_model, save_path)
    print(f"   ✅ Saved Model  → {save_path}")
    print(f"   ✅ Saved Report → {report_path}")

print("\nBaseline Training Complete.")
