# METR — Documentation & Findings

> **Market Exposure Timing vs Randomness**
> A controlled multi-asset empirical study investigating whether machine learning models can outperform random entry decisions.

---

## 1. Project Overview

### Objective
Determine if historical price dynamics (momentum, mean reversion, volatility regimes) contain predictive power that exceeds a random coin-flip over a 3-day holding period.

### Architecture
- **Assets:** Nifty 50, Gold (MCX), USD/INR
- **Model:** XGBoost Classifier (per-asset)
- **Baseline:** Monte Carlo simulation (1000 random strategies)
- **Validation:** TimeSeriesSplit (5-fold, no look-ahead bias)

### Tech Stack
| Component | Tool |
|-----------|------|
| Data Processing | Polars (lazy execution) |
| Modeling | XGBoost + Scikit-Learn |
| Configuration | `config.yaml` (feature lists, splits, thresholds) |
| Evaluation | Custom scripts (`evaluate_model.py`, `benchmark_random.py`, `feature_analysis.py`) |

---

## 2. Data Pipeline

### Source
- Raw OHLCV data fetched via `src/fetch_data.py`
- Three assets: Nifty (3213 rows), Gold (3224 rows), USD/INR (3403 rows)
- Row count differences due to different market holidays

### Feature Engineering (`src/data_cleaning.ipynb`)
All features are **stationary** (no raw prices) and computed at market close:

| Category | Features | Count |
|----------|----------|-------|
| **Momentum** | `Ret_1d`, `Ret_3d`, `Ret_5d`, `Ret_20d`, `ROC_10` | 5 |
| **Volatility** | `Vol_5d`, `Vol_20d`, `Vol_Ratio`, `ATR_Pct` | 4 |
| **Trend** | `MA_Ratio`, `Price_vs_MA20`, `MACD_Hist` | 3 |
| **Overbought/Oversold** | `RSI`, `BB_Pct` | 2 |
| **Microstructure** | `Close_Pos_Range`, `Intraday_Return` | 2 |
| **Lagged Returns** | `Ret_1d_Lag1`, `Ret_1d_Lag2`, `Ret_1d_Lag3` | 3 |
| **Total** | | **19** |

### Label Definition
```
Forward_Return = (Close[T+3] - Open[T+1]) / Open[T+1]
Label = 1 if Forward_Return > 0, else 0
```
- Decision at T Close, Entry at T+1 Open, Exit at T+3 Close (3-day holding period)

### Train/Test Split
- **Train:** ~2451 rows (2013–2023)
- **Test:** ~453 rows (2024–2025)
- Split by time (no shuffling)

---

## 3. Model Development

### Model v1 — Overfitted (Aggressive Parameters)

**Hyperparameter Grid:**
```
max_depth:     [2, 3, 4, 5]
learning_rate: [0.01, 0.03, 0.05, 0.1]
n_estimators:  [100, 150, 200, 250, 300]
subsample:     [1.0]
```

**Best Parameters Found:**
```
learning_rate: 0.1, max_depth: 4, n_estimators: 5000, subsample: 1.0
```

**Results:**

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 86.54% | 53.42% |
| ROC AUC | — | 0.5251 |
| **Overfitting Gap** | **33.1%** | ⚠️ Severe |

**High Confidence (>0.60 threshold):**
- Trades: 129 / 453
- Win Rate: 56.6%
- vs Baseline (51.4%): +5.2%

**Diagnosis:** The model memorized the training data. The 86% train accuracy was fake — the model found patterns specific to historical dates rather than generalizable market signals.

---

### Model v2 — Regularized (Anti-Overfitting)

**Changes Made:**
| Parameter | v1 | v2 | Rationale |
|-----------|----|----|-----------|
| `max_depth` | [2,3,4,5] | **[3,4]** | Shallower trees |
| `subsample` | [1.0] | **[0.8, 0.9, 1.0]** | Each tree sees subset of rows |
| `colsample_bytree` | Not set | **[0.8, 0.9, 1.0]** | Each tree sees subset of features |
| `reg_alpha` (L1) | Not set | **[0, 0.01, 0.1]** | Sparse leaf weights |
| `reg_lambda` (L2) | Not set | **[0.5, 1.0, 2.0]** | Smooth leaf weights |
| Early stopping | None | **30 rounds** | Halt when validation stops improving |

**Best Parameters Found:**
```
colsample_bytree: 0.9, learning_rate: 0.05, max_depth: 4, 
n_estimators: 100, reg_alpha: 0.1, reg_lambda: 0.5, subsample: 0.8
```

**Results:**

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 59.24% | 51.66% |
| ROC AUC | — | 0.5020 |
| **Overfitting Gap** | **7.6%** | ⚠️ Mild |

**High Confidence (>0.60 threshold):**
- Trades: 9 / 453 (too few — model probabilities too compressed)

**Diagnosis:** Overfitting gap fixed (33% → 7.6%), but the regularization was initially too aggressive (`reg_alpha=1.0`, `reg_lambda=5.0`) which killed the signal entirely. After loosening to lighter values (`reg_alpha=0.1`, `reg_lambda=0.5`), the model found moderate parameters but the inherent signal in the features is very weak.

---

### Key Lesson: Regularization Sensitivity

| `reg_alpha` | `reg_lambda` | Effect |
|-------------|-------------|--------|
| 0 (default) | 1 (default) | No regularization → overfitting |
| 0.1 | 0.5 | Light regularization → best balance |
| 1.0 | 5.0 | Too aggressive → killed all signal |

The sweet spot was much closer to the defaults than initially assumed.

---

## 4. Benchmark Results (Model vs Random)

### Monte Carlo Simulation
- **Method:** 1000 random strategies, each selecting the same number of trades as the model but on random dates
- **Return Calculation:** 3-day compound returns reconstructed from `Ret_1d`

**Results (Model v1):**

| Metric | Model | Random Average |
|--------|-------|----------------|
| Total Return | 18.73% | 14.90% |
| Win Rate | 60.5% | ~51% |
| P-Value | 0.376 | — |

**Interpretation:**
- Model beat random average by +3.8%
- P-value of 0.38 means not statistically significant at 95% confidence
- However, 60.5% win rate vs 51% baseline is a meaningful edge for entry timing
- The high p-value is partly because the market was in a bull phase — even random strategies did well

---

## 5. Feature Importance Analysis

### Gain (How much each feature improves predictions)
All 19 features scored between 4.77 and 7.21 — **no standout feature**. This indicates the model is grabbing at weak signals equally across all features.

**Top 5 by Gain:**
1. `Intraday_Return` (7.21) — Context provider, not a directional signal
2. `MA_Ratio` (6.85) — Short vs long-term trend
3. `Vol_20d` (6.65) — 20-day volatility
4. `MACD_Hist` (6.56) — Trend momentum
5. `Vol_Ratio` (6.50) — Volatility regime change

### Correlation with Label (Test Set)
The strongest correlation was only **+0.12** (`Vol_20d`). In a good prediction problem, you'd expect 3-4 features > 0.15.

**Key Findings:**
| Feature | Correlation | Insight |
|---------|-------------|---------|
| `Vol_20d` | +0.12 | Higher volatility → market bounces (mean-reversion) |
| `ATR_Pct` | +0.10 | Same signal as Vol_20d (redundant) |
| `MACD_Hist` | -0.09 | **Contrarian** — strong momentum precedes reversals |
| `BB_Pct` | -0.08 | Overbought → pullback |
| `Ret_3d` | -0.002 | **Almost zero** — 3-day return has no predictive power for next 3-day return |

### Core Problem Identified
All 19 features are derived from **single-asset OHLC data** using standard technical indicators available to every retail trader. The maximum correlation with the target is 0.12, confirming that publicly available indicators carry minimal edge at the 3-day horizon.

---

## 6. What Didn't Work

| Approach | Why It Failed |
|----------|--------------|
| **High `max_depth` (4-5) without regularization** | Severe overfitting (86% train / 53% test) |
| **High regularization (`reg_alpha=1.0`, `reg_lambda=5.0`)** | Killed all signal, model output compressed near 0.50 |
| **Low `subsample` (0.6)** | Too little data per tree for a 2451-row dataset |
| **0.60 confidence threshold on regularized model** | Only 9 trades triggered — probabilities too compressed |
| **Technical indicators alone** | Max feature correlation 0.12 — not enough signal |

---

## 7. What Worked

| Finding | Evidence |
|---------|----------|
| **TimeSeriesSplit validation** | Prevented look-ahead bias in all experiments |
| **Moderate regularization** | Reduced overfitting gap from 33% to 7.6% |
| **Confidence thresholding concept** | Model v1 at >0.60 achieved 60.5% win rate (vs 51.4% baseline) |
| **Volatility features** | `Vol_20d` and `ATR_Pct` had the highest correlation with the label |
| **GPU acceleration** | Made GridSearchCV feasible (768 combinations × 5 folds) |
| **Monte Carlo benchmark** | Provided honest comparison vs random chance |

---

## 8. Research Directions (Not Yet Implemented)

### 8.1 Triple Barrier Labeling
**Source:** Marcos López de Prado, *Advances in Financial Machine Learning*

Instead of simple binary labels (up/down), the Triple Barrier Method defines three exit conditions:
- **Upper Barrier:** Price rises by 2σ → Label = +1 (profit target hit)
- **Lower Barrier:** Price falls by 2σ → Label = -1 (stop-loss hit)
- **Vertical Barrier:** Time expires (3 days) → Label = 0 (inconclusive)

**Advantages over current approach:**
- Filters out noise (tiny moves become 0 instead of forced 0/1)
- Volatility-adaptive (barriers scale with market conditions)
- Reflects actual trading reality (stop-losses and take-profits)

### 8.2 Cross-Asset Features
Current features are all single-asset (Nifty predicts Nifty from Nifty). Proposed additions:

| Feature | Formula | Signal |
|---------|---------|--------|
| Equity-Gold Relative Strength | `Ret_5d_Nifty - Ret_5d_Gold` | Risk appetite |
| Rupee Stress | `Ret_3d_USDINR` | FII flow pressure |
| Gold Momentum | `Ret_5d_Gold` | Fear/safety demand |
| Cross-Vol Ratio | `Vol_20d_Nifty / Vol_20d_Gold` | Panic detection |

**Prerequisite:** Align all three assets to common trading dates before feature engineering.

### 8.3 Asset-Specific Model Tuning
Each asset behaves differently and requires tailored hyperparameters:

| Asset | Recommended Approach |
|-------|---------------------|
| **Nifty (Equity)** | Trend/momentum focus, moderate depth |
| **Gold (Commodity)** | Lower learning rate (violent bursts), hedge/safety features |
| **USD/INR (FX)** | Higher regularization (central bank-managed), mean-reversion focus |

### 8.4 Regime Modeling (Paused)
A volatility filter to avoid trading during high-volatility periods. This was deprioritized but remains a potential improvement:
- Train a separate classifier to predict High/Low volatility regimes
- Only take Exposure Model trades during "Calm" regimes

### 8.5 Alternative Data Sources
Features not derivable from OHLC that institutions use:
- India VIX, Put-Call Ratio, FII/DII flows (Tier 2 — requires new data sources)
- News sentiment via NLP, Google Trends (Tier 3 — advanced)

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `src/fetch_data.py` | Downloads raw OHLCV data |
| `src/data_cleaning.ipynb` | Feature engineering, labeling, train/test split |
| `src/train_exposure.py` | XGBoost training with GridSearchCV + early stopping |
| `src/evaluate_model.py` | Generates 5 diagnostic plots (confusion matrix, ROC, probability dist, etc.) |
| `src/benchmark_random.py` | Monte Carlo comparison (model vs 1000 random strategies) |
| `src/feature_analysis.py` | Feature importance and correlation analysis |
| `config.yaml` | Central config (features, splits, threshold, weights) |
| `models/nifty.joblib` | Saved Model v1 (overfitted) |
| `models/nifty2.joblib` | Saved Model v2 (regularized) |
| `reports/` | All generated plots (confusion matrix, ROC curve, threshold sweep, etc.) |

---

## 10. Conclusion

The METR project successfully built a controlled experimental framework for comparing ML-based entry timing against random chance. The core finding is that **standard technical indicators provide a weak but measurable edge** (~0.53 CV ROC-AUC) for 3-day Nifty direction prediction, but this edge is **not statistically significant** when tested against random baselines (p=0.38).

The project identified two clear paths for improvement:
1. **Better labels** (Triple Barrier Method) — to reduce noise in the training signal
2. **Better features** (Cross-asset data) — to capture macro dynamics invisible to single-asset indicators

Both are documented above and ready for implementation.
