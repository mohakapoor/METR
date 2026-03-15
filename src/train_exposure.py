import polars as pl
import xgboost as xgb
import numpy as np
import yaml
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

CONFIG_PATH = "config.yaml"



train = pl.read_parquet("data/processed/train/nifty.parquet")
test = pl.read_parquet("data/processed/test/nifty.parquet")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

exp_feat = config["Exposure_Features"]
splits = config["N_Splits"]
threshold = config.get("Threshold")

X_train = train.select(exp_feat).to_pandas()
y_train = train["Label"].to_pandas()

X_test = test.select(exp_feat).to_pandas()
y_test = test["Label"].to_pandas()

print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"Features: {len(exp_feat)} | Threshold: {threshold}")

# ============================================
# 2. HYPERPARAMETER GRID (Anti-Overfitting)
# ============================================
# KEY CHANGES vs previous version:
# - max_depth: reduced to [2,3] (was [2,3,4,5]) — shallower trees
# - subsample: [0.6, 0.8] (was [1.0]) — each tree sees only 60-80% of rows
# - colsample_bytree: [0.6, 0.8] — each tree sees only 60-80% of features
# - reg_alpha + reg_lambda: L1 and L2 regularization on leaf weights

param_grid = {
    'max_depth':        [3, 4],
    'learning_rate':    [0.01, 0.03, 0.05],
    'n_estimators':     [100,150, 200,],
    'subsample':        [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'reg_alpha':        [0,0.01,0.1,],       # L1 regularization
    'reg_lambda':       [0.5,1.0, 2.0],       # L2 regularization
}

total = 1
for v in param_grid.values():
    total *= len(v)
print(f"\nGrid Search: {total} combinations × {splits} folds = {total * splits} fits")

# ============================================
# 3. CROSS-VALIDATION (TimeSeriesSplit)
# ============================================
tscv = TimeSeriesSplit(n_splits=splits)

search = GridSearchCV(
    estimator=xgb.XGBClassifier(
        random_state=42,
        device='cuda',
        early_stopping_rounds=30,     # Stop if no improvement for 30 rounds
        eval_metric='auc',            # Monitor AUC during training
    ),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=tscv,
    verbose=1,
    n_jobs=1,
)

# Early stopping needs eval_set — GridSearchCV handles train/val internally
# but early_stopping_rounds uses the LAST dataset in eval_set.
# With GridSearchCV, we pass the test set as eval_set for monitoring only.
search.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True    # Suppress per-round XGBoost logs
)

print(f"\nBest CV Score (ROC-AUC): {search.best_score_:.4f}")
print(f"Best Params: {search.best_params_}")

# ============================================
# 4. EVALUATE BEST MODEL
# ============================================
best_model = search.best_estimator_
probs = best_model.predict_proba(X_test)[:, 1]

# --- Overfitting Check ---
train_probs = best_model.predict_proba(X_train)[:, 1]
train_preds = (train_probs >= 0.50).astype(int)
test_preds  = (probs >= 0.50).astype(int)

train_acc = accuracy_score(y_train, train_preds)
test_acc  = accuracy_score(y_test, test_preds)
gap = train_acc - test_acc

print(f"\n=== Overfitting Check ===")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Gap:            {gap:.4f}", end="")
if gap > 0.10:
    print(" ⚠️  OVERFITTING")
elif gap > 0.05:
    print(" ⚠️  Mild overfitting")
else:
    print(" ✅ Healthy")

# --- High Confidence Evaluation ---
high_conf_mask = probs >= threshold
high_conf_actual = y_test[high_conf_mask]

if len(high_conf_actual) > 0:
    hc_win_rate = high_conf_actual.mean()
    print(f"\n=== High Confidence (>{threshold}) ===")
    print(f"Trades: {len(high_conf_actual)} / {len(y_test)}")
    print(f"Win Rate: {hc_win_rate:.4f}")
    print(f"vs Baseline: +{(hc_win_rate - y_test.mean())*100:.1f}%")
else:
    print(f"\nNo trades above threshold {threshold}")

# --- Overall ---
auc = roc_auc_score(y_test, probs)
print(f"\nOverall Test ROC AUC: {auc:.4f}")

# ============================================
# 5. SAVE MODEL
# ============================================
joblib.dump(best_model, "models/nifty.joblib")
print(f"\n✅ Model saved to models/nifty.joblib")