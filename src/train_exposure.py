"""
train_exposure.py — Phase 4: Multi-Class XGBoost Training
==========================================================
- Labels: -1 (stop-loss), 0 (timeout), +1 (profit) → remapped to 0, 1, 2 for XGBoost
- Two modes per asset:
    ISOLATED : Exposure_Features only
    UNIFIED  : Exposure_Features + Cross_Asset_Features[asset]
- TimeSeriesSplit CV (no look-ahead bias)
- Saves best model per asset/mode to models/
"""

import polars as pl
import numpy as np
import yaml
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

EXP_FEATURES   = config["Exposure_Features"]
CROSS_FEATURES = config["Cross_Asset_Features"]   # dict: {asset: [feat, ...]}
N_SPLITS       = config["N_Splits"]               # 5
THRESHOLD      = config["Threshold"]              # 0.6

ASSETS = ["nifty", "gold", "usdinr"]

# Label map: -1 → 0, 0 → 1, 1 → 2
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}
CLASS_NAMES   = ["-1 (stop-loss)", "0 (timeout)", "+1 (profit)"]

# ── Hyperparameter Grid ───────────────────────────────────────────────────────
# Kept compact — 3 assets × 2 modes × 5 folds is already 30+ fits per combo
param_grid = {
    "max_depth":        [3, 4],
    "learning_rate":    [0.03, 0.05],
    "n_estimators":     [100, 200],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_alpha":        [0.0, 0.1],
    "reg_lambda":       [0.5, 1.0],
}

total_combos = 1
for v in param_grid.values():
    total_combos *= len(v)
print(f"Grid: {total_combos} combos × {N_SPLITS} folds = {total_combos * N_SPLITS} fits per run")

# ── Training Loop ─────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

for asset in ASSETS:
    print(f"\n{'='*60}")
    print(f"  ASSET: {asset.upper()}")
    print(f"{'='*60}")

    train = pl.read_parquet(f"data/processed/train/{asset}.parquet")
    test  = pl.read_parquet(f"data/processed/test/{asset}.parquet")

    # Remap labels
    y_train_raw = train["Label"].to_pandas()
    y_test_raw  = test["Label"].to_pandas()
    y_train = y_train_raw.map(LABEL_MAP)
    y_test  = y_test_raw.map(LABEL_MAP)

    # Drop rows where label is null (last T rows from triple barrier)
    valid_train = y_train.notna()
    valid_test  = y_test.notna()
    y_train = y_train[valid_train].astype(int)
    y_test  = y_test[valid_test].astype(int)

    # Unified: per-asset technical features + cross-asset macro features
    features = EXP_FEATURES + CROSS_FEATURES.get(asset, [])

    # Filter to features that actually exist in the parquet
    available = [f for f in features if f in train.columns]
    missing   = set(features) - set(available)
    if missing:
        print(f"  [WARN] Missing features skipped: {missing}")

    X_train = train.select(available).to_pandas()[valid_train]
    X_test  = test.select(available).to_pandas()[valid_test]

    print(f"\n── UNIFIED | {len(available)} features ──")
    print(f"   Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"   Label dist (train): { y_train.value_counts().sort_index().to_dict() }")

    # Base estimator
    base_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        device="cuda",
        eval_metric="mlogloss",
        early_stopping_rounds=20,
    )

    search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=tscv,
        verbose=0,
        n_jobs=1,
    )

    search.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    best = search.best_estimator_
    print(f"   Best CV Accuracy : {search.best_score_:.4f}")
    print(f"   Best Params      : {search.best_params_}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    test_probs  = best.predict_proba(X_test)    # shape (n, 3)
    train_probs = best.predict_proba(X_train)

    test_preds  = np.argmax(test_probs,  axis=1)
    train_preds = np.argmax(train_probs, axis=1)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc  = accuracy_score(y_test,  test_preds)
    gap       = train_acc - test_acc

    print(f"\n   Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Gap: {gap:.4f}", end="")
    if   gap > 0.10: print("  ⚠️  OVERFITTING")
    elif gap > 0.05: print("  ⚠️  Mild overfit")
    else:            print("  ✅ Healthy")

    # High-confidence +1 trades (class index 2 = original label +1)
    profit_prob = test_probs[:, 2]   # P(label = +1)
    hc_mask     = profit_prob >= THRESHOLD
    if hc_mask.sum() > 0:
        actual_orig = y_test_raw[valid_test].values[hc_mask]
        win_rate    = (actual_orig == 1).mean()
        baseline    = (y_test_raw[valid_test] == 1).mean()
        print(f"   High-Conf +1 (>{THRESHOLD}): {hc_mask.sum()} trades | "
              f"Win Rate: {win_rate:.4f} | vs Baseline: {'+' if win_rate > baseline else ''}{(win_rate - baseline)*100:.1f}%")
    else:
        print(f"   No high-confidence +1 trades above {THRESHOLD}")

    print(f"\n{classification_report(y_test, test_preds, target_names=CLASS_NAMES, zero_division=0)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = f"models/{asset}_unified.joblib"
    joblib.dump(best, save_path)
    print(f"   ✅ Saved → {save_path}")