# METR — Documentation & Findings

> **Market Exposure Timing vs Randomness**
> A controlled multi-asset empirical study investigating whether machine learning models can outperform random entry decisions.

---

## 1. Project Overview

### Objective
Determine if machine learning models can identify "high-conviction" regimes for a primary momentum signal. We use **Meta-Labeling** to filter out noise and improve the precision of short-term market entries.

### Architecture
- **Layer 1 (Signal)**: Primary 5-day momentum signal.
- **Layer 2 (Filter)**: Binary XGBoost Classifier (Meta-Model).
- **Assets:** Nifty 50, Gold (MCX), USD/INR
- **Baseline:** Logistic Regression + Scaling
- **Validation:** TimeSeriesSplit (5-fold, no look-ahead bias)

### Tech Stack
| Component | Tool |
|-----------|------|
| Data Processing | Polars (lazy execution) |
| Modeling | XGBoost + Scikit-Learn |
| Configuration | `config.yaml` (feature lists, splits, thresholds) |
| Data Source | `yfinance` |
| Evaluation | Custom scripts (`evaluate_model.py`, `benchmark_random.py`, `feature_analysis.py`) |

---

## 2. Data Pipeline

### Source
- Raw OHLCV data fetched via `src/fetch_data.py`
- Assets: Nifty, Gold, USD/INR, and **India VIX** (implied volatility / fear gauge)
- Row count differences due to different market holidays

### Data Pipeline Stages
The pipeline is split into two distinct specialized notebooks:

1. **Alignment (`src/data_cleaning.ipynb`)**:
    - **Asset Joining**: Programmatic merging of Nifty, Gold, USD/INR, and VIX into a unified timeframe.
    - **Holiday Synchronization**: Stabilizing the dataset against non-overlapping market holidays (e.g., MCX vs. NSE).
    - **Lookback Buffer**: Anchoring the training dataset at 2014-01-01 while preserving 2012-2013 for feature warm-up.
    - **Date Intersection**: Ensuring all cross-asset features are calculated on common trading days (3,446 days total).

2. **Synthesis (`src/feature_engineering.ipynb`)**:
    - **Memory Persistence**: Application of **Fractional Differentiation** (orders $d \in [0.30, 0.45]$) to preserve 77-91% of historical memory while ensuring stationarity.
    - **Macro Indicators**: Integration of the **India VIX** (implied volatility) as a forward-looking fear gauge.
    - **Labeling**: Generating the **Triple Barrier** target (-1, 0, +1) using per-asset volatility-adaptive thresholds ($k$) and a **5-day window** ($T$).
    - **Meta-Labeling**: Synthesizing the `Meta_Label` (1 if Signal = Win, 0 if Signal = Fail/Timeout).
    - **Inter-Asset Dynamics**: Creation of cross-asset features (RS, Risk-Off, FX Sensitivity).

**Total Features:** 13 (Optimized) + 7 (Macro and Cross-Asset).

### Meta-Label Definition
```
Signal = 1 if Ret_5d > 0 else -1
Meta_Label = 1 if (Signal == TB_Label) else 0
```
- **TB_Label**: Result of the Triple Barrier over 5 sessions.

### Train/Test Split
- **Buffer Data:** 2012–2013 (Used for rolling indicators and Fractional Differentiation lookbacks)
- **Train (Anchor):** 2014–2023 (~2200 rows)
- **Test:** 2024–2025 (~450 rows)
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

## 8. Finalized Labeling Approach: Triple Barrier Method
*(Calculated in `src/feature_engineering.ipynb`)*

**Source:** Marcos López de Prado, *Advances in Financial Machine Learning*

Instead of simple binary labels (up/down), a forward-scanning Triple Barrier Method is utilized. This defines three exit conditions over a rolling window $T$:
- **Upper Barrier:** Price rises by $k\sigma$ → Label = +1 (profit target hit first)
- **Lower Barrier:** Price falls by $k\sigma$ → Label = -1 (stop-loss hit first)
- **Vertical Barrier:** Time $T$ expires → Label = 0 (inconclusive / timeout)

Where $\sigma$ is the daily volatility (computed as the rolling 20-day standard deviation of daily returns).

### 8.1 In-Depth Labeling Logic
Traditional binary labeling (Up/Down) forces a model to guess direction regardless of whether the move was meaningful or just noise. **METR** uses a volatility-adaptive sequence:

Entry: $P_0$ (at Open of $T+1$)  
Upper Barrier: $P_0 \cdot (1 + k\sigma)$  
Lower Barrier: $P_0 \cdot (1 - k\sigma)$  
Vertical Barrier: Time $T$ expires → Label = 0 (inconclusive / timeout)

#### Rationale for $T=5$
Empirically, $T=3$ proved too noisy for stable directional learning. By extending to $T=5$, the **Return Spread** between profit and loss labels increases significantly across all three assets, providing a cleaner signal for the Meta-Model to filter.

#### Asset-Specific Scalings ($k$)
- **NIFTY**: $k=1.5$
- **GOLD**: $k=1.75$ (Higher volatility tail)
- **USDINR**: $k=1.5$

---

## 9. Advanced Feature Engineering: Fractional Differentiation
*(Implemented in `src/feature_engineering.ipynb`)*

To solve the stationarity-memory trade-off, **Fractional Differentiation** ($d \in [0.1, 0.9]$) was implemented. This ensures the features are stationary for ML models while retaining as much historical "memory" as possible, unlike standard integer-differencing ($d=1$).

### 9.1 Methodology
The study uses `src/frac_diff.py` with a threshold of **$10^{-4}$** to balance mathematical precision with data availability (lookback length). Instead of a strict ADF $p < 0.05$ cutoff, parameters were selected by maximizing **Memory Preservation** (correlation with original series) while achieving "good enough" stationarity.

### 9.2 Final Manual Selection ($d$)
| Asset | Optimal $d$ | Correlation | ADF p-value | Characteristic |
|---|---|---|---|---|
| **Nifty 50** | **0.45** | 0.818 | 0.079 | High memory / Strong trend persistence |
| **Gold** | **0.50** | 0.772 | 0.096 | Best memory-stationarity compromise |
| **USD/INR** | **0.30** | 0.908 | 0.034 | Perfectly stationary with 91% memory |

---

## 10. Feature Dictionary (In-Depth)

The following features are synthesized in `src/feature_eng.py` and `src/feature_engineering.ipynb`:

### 10.1 Momentum & Price Action
- **`Ret_1d, Ret_3d, Ret_5d, Ret_20d`**: Standard log-returns for capturing multi-timeframe momentum trends.
- **`Intraday_Return`**: `(Close - Open) / Open`. Measures within-session conviction; often identifies institutional accumulation during choppy sessions.
- **`Gap`**: `(Open - Prev_Close) / Prev_Close`. Captures overnight sentiment shifts and sensitivity to global market moves.
- **`Close_Pos_Range`**: `(Close - Low) / (High - Low)`. Pinpoints "pin-bars" and price rejection at extremes. Values >0.8 indicate bullish rejection of low prices.
- **`Range_Expansion`**: `(High - Low) / (Prev_High - Prev_Low)`. A quick-response volatility spike indicator.

### 10.2 Volatility & Regime Detection
- **`Vol_Ratio`**: `Vol_5d / Vol_20d`. Detects when short-term volatility is expanding relative to the monthly baseline—a primary predictor of trend reversals.
- **`ATR_Pct`**: `Average True Range / Price`. Normalizes risk across time, allowing the model to compare volatility in low-price vs. high-price regimes.
- **`BB_Pct`**: Bollinger Band %B. Quantifies where price sits relative to its 20-day standard deviation bands. Excellent for identifying over-extended mean-reversion setups.

### 10.3 Stationarity & Memory (FracDiff)
- **`FD_Close`**: Fractionally Differentiated Close. The crown jewel of the logic; it maintains the underlying trend signal (memory) while achieving statistical stationarity (as verified by ADF tests).
- **`FD_Close_Lag1`**: Catching the first derivative of the stationary series to identify momentum in a "safe" (non-integrated) space.

### 10.4 Cross-Asset Dynamics (Phase 3)
- **`Relative Strength (RS)`**: `Gold_Ret_5d - Nifty_Ret_5d`. Measures risk-appetite shifts. When Gold leads Nifty, capital is often fleeing to safety.
- **`Usdinr_Ret_x`**: Currency returns as a macro-economic pressure gauge. High USDINR volatility often signals FII (Foreign Institutional Investor) outflow.
- **`Momentum_Align`**: Checks if the individual asset's return sign matches the broader benchmark—identifying "true" strength vs. "lucky" market-wide drifts.
- **`Risk_Off` Indicator**: Binary flag (1 if Gold Leads, 0 otherwise).
- **`Equity_Stress` Indicator**: Binary flag (1 if Nifty 5d-return < 0).

---

## 11. Project Evolution: Phase 6 Meta-Filtering ✅ *(In Progress)*

The project has pivoted from raw directional prediction to a sophisticated **Meta-Labeling** architecture.

- **Baseline Established**: A Logistic Regression baseline proved that the relationship between technicals and "Momentum Quality" is non-linear.
- **Nifty Benchmark**: The linear model fails to cross the 52% probability threshold on Nifty, establishing a "Zero-Recall" benchmark for XGBoost to beat.
- **Feature Separation**: Identified that **Relative Strength vs Gold** is the strongest "Green Light" for Nifty momentum success.
- **Forward-Looking Volatility**: Transitioned all models to utilize **India VIX** as the primary regime detector.

---

## 12. Future Research Directions

### 12.1 Asset-Specific Model Tuning
Each asset behaves differently and requires tailored hyperparameters:

| Asset | Recommended Approach |
|-------|---------------------|
| **Nifty (Equity)** | Trend/momentum focus, moderate depth |
| **Gold (Commodity)** | Lower learning rate (violent bursts), hedge/safety features |
| **USD/INR (FX)** | Higher regularization (central bank-managed), mean-reversion focus |

### 12.2 Regime Modeling
A volatility filter to avoid trading during high-volatility periods. This remains a potential improvement:
- Train a separate classifier to predict High/Low volatility regimes.
- Only take Exposure Model trades during "Calm" regimes.

### 12.3 Multi-Class & Meta-Labeling
Transitioning from binary -1, 0, +1 labeling to a model that can predict the *specific barrier hit type* with higher confidence, potentially using Meta-Labeling (as described by de Prado) to filter out false positives.

---

## 13. File Reference

| File | Purpose |
|------|---------|
| `src/fetch_data.py` | Automated historical data ingestion (yfinance) |
| `src/data_cleaning.ipynb` | Join, align, and synchronize multi-asset timestamps |
| `src/feature_engineering.ipynb` | Final synthesized dataset generation and Meta-Labeling |
| `src/feature_eng.py` | Technical indicator library (Polars-optimized) |
| `src/tripple_barrier.py` | Algorithm for forward-scanning volatility labeling |
| `src/train_trade_filter.py` | XGBoost Meta-Model training (GPU accelerated) |
| `src/train_baseline.py` | Logistic Regression baseline model |
| `src/feature_separation.py` | ROC AUC analysis of individual feature signal strength |
| `data/processed/train/` | Cleaned 2014-2023 training shards |
| `data/processed/test/` | Out-of-sample 2024-2025 evaluation sets |

---

## 13. Conclusion

The METR project successfully built a controlled experimental framework for comparing ML-based entry timing against random chance. The core finding is that **standard technical indicators provide a weak but measurable edge** (~0.53 CV ROC-AUC) for 3-day Nifty direction prediction, but this edge is **not statistically significant** when tested against random baselines (p=0.38).

The project identified two clear paths for improvement:
1. **Better labels** (Triple Barrier Method) — to reduce noise in the training signal
2. **Better features** (Cross-asset data) — to capture macro dynamics invisible to single-asset indicators

Both are documented above and ready for implementation.

---
