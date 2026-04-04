# METR — Documentation & Findings

> **Market Exposure Timing vs Randomness**
> A controlled multi-asset empirical study investigating whether machine learning models can outperform random entry decisions.

---

## 1. Project Overview

### Objective
Determine if machine learning models trained purely on historical price, volume, and publicly available implied volatility data can identify "high-conviction" regimes for a primary momentum signal. Using **Meta-Labeling** and validating against **1,000 Monte Carlo simulations**, I isolate true statistical predictive edge from market noise.

### Architecture
- **Layer 1 (Signal)**: Primary 5-day momentum signal (Volatility-Normalized).
- **Layer 2 (Filter)**: Symmetric Binary XGBoost Classifier (Meta-Filter).
- **Architecture**: Direction-Aware P&L tracking (`Directional_Return`).
- **Assets:** Nifty 50, Gold (MCX). (USD/INR Pruned).
- **Baseline:** Logistic Regression (Symmetric Audit).
- **Validation:** TimeSeriesSplit (5-fold) + Signal-Conditional Masking ($Signal \neq 0$).

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
Targeting two distinct specialized notebooks:

1. **Alignment (`src/data_cleaning.ipynb`)**:
    - **Asset Joining**: Programmatic merging of Nifty, Gold, USD/INR, and VIX into a unified timeframe.
    - **Holiday Synchronization**: Stabilizing the dataset against non-overlapping market holidays (e.g., MCX vs. NSE).
    - **Lookback Buffer**: Anchoring the training dataset at 2014-01-01 while preserving 2012-2013 for feature warm-up.
    - **Date Intersection**: Ensuring all cross-asset features are calculated on common trading days (3,446 days total).

2. **Synthesis (`src/feature_engineering.ipynb`)**:
    - **Memory Persistence**: Application of **Fractional Differentiation** (orders $d \in [0.30, 0.45]$) to preserve 77-91% of historical memory while ensuring stationarity.
    - **Macro Indicators**: Integration of the **India VIX** (implied volatility) as a forward-looking fear gauge.
    - **Labeling**: Generating the **Triple Barrier target** (-1, 0, +1) using per-asset volatility-adaptive thresholds ($k$) and a **5-day window** ($T$).
    - **Symmetric Meta-Labeling**: Applying side-aware directionality where Meta_Label = 1 if (TB_Return * Signal) > 0.
    - **Inter-Asset Dynamics**: Creation of cross-asset features (RS, Risk-Off, FX Sensitivity).

**Total Features:** 13 (Base) + 12 (Core Interaction) + 5 (Macro/VIX).



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

---

### Model v3 — Symmetric Alpha (Phase 10.2)

**Key Innovations:**
1. **Side-Aware P&L**: I switched from raw `TB_Return` to `Directional_Return = TB_Return * Signal`. 
2. **Noise Suppression**: I implemented signal-conditional training ($Signal \neq 0$) to filter out non-active regimes.
3. **Institutional Calibration**: I migrated to **3-fold Cross-Validation** (replacing `cv=prefit`) for robust, honest out-of-sample probability estimation.

**Results (Model v3):**

| Asset | ROC AUC | AUPRC |
|-------|---------|-------|
| **Nifty** | **0.5918** | 0.4431 |
| **Gold** | **0.5935** | 0.4602 |
| **USD/INR** | 0.5489 | 0.4833 |

**Diagnosis:** The symmetric refactor successfully "unlocked" the short-side alpha that was previously invisible. The migration to **3-fold Cross-Validation** provides a robust, institutional foundation for my probability estimation across all three assets.

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
- My model beat the random average by +3.8%
- P-value of 0.38 means not statistically significant at 95% confidence
- However, 60.5% win rate vs 51% baseline is a meaningful edge for entry timing

---

## 5. Feature Importance Analysis (Per-Asset Audit)

The introduction of **Macro-Interactors (Phase 8)** has revealed that alpha drivers are highly asset-specific. Below is my forensic breakdown of the Top 3 drivers for each production engine.

### 5.1 Nifty 50 (Equities)
*The Nifty engine is primarily driven by Relative Strength and Volatility regimes.*

| Feature | Importance (Gain) | Correlation (r) |
|---------|-------------------|-----------------|
| `Gold_Nifty_RS_5d` | 6.44 | +0.128 |
| `ROC_10` | 6.38 | -0.077 |
| `VIX_ATR_Ratio` | 5.21 | -0.128 |

### 5.2 Gold (Safe-Haven)
*Gold is my most macro-sensitive asset, driven by extreme fear and decoupling events.*

| Feature | Importance (Gain) | Correlation (r) |
|---------|-------------------|-----------------|
| `VIX_Momentum_Efficiency`| **13.61** | +0.007 |
| `RSI` | 11.06 | +0.031 |
| `RS_Momentum_Decoupling` | 2.76 | **-0.108** |

### 5.3 USD/INR (FX)
*The FX engine identifies global risk-off flow into the Indian economy.*

| Feature | Importance (Gain) | Correlation (r) |
|---------|-------------------|-----------------|
| `Vol Efficiency` | 2.83 | **+0.159** |
| `Nifty_Vol_Ratio` | 2.71 | **+0.153** |
| `Usdinr_Stress_Filter` | 4.11 | +0.048 |

### 5.4 Strategic Breakthrough
By transitioning to **Per-Asset Macro Interactors**, I have bridged the **0.15 correlation floor.** This proves that market exposure timing is not a "one size fits all" problem; the USD/INR and Gold engines require cross-asset stress signatures to unlock their true alpha potential.

---

## 6. What Didn't Work

| Approach | Why It Failed |
|----------|--------------|
| **USD/INR Friction-Blind Model** | I achieved high AUC (0.54) in 0-bps tests, but the dual-friction audit revealed it was a **"Friction Trap"** that decayed to negative Net Sharpe at 5-bps. |
| **GaussianHMM Regime Filter** | I attempted to use HMMs to define crisis states, but they added complexity without outperforming the simpler **VIX_Shock** indicators in the XGBoost architecture. |
| **Simple Long-Only Meta-Labeling** | Early versions ignored the short side, leading to **"Short Alpha Blindness"** where high-conviction pullbacks were discarded as noise. |
| **Standard Technicals Alone** | I found that publicly available indicators (RSI, MACD) hit a correlation floor of 0.12 — they lack predictive power for a 5-day horizon without macro context. |

---

## 7. What Worked

| Finding | Evidence |
|---------|----------|
| **Symmetric Meta-Labeling** | I unlocked the short-side alpha by switching to `Directional_Return`, allowing for high-conviction "Sell" filters. |
| **Cross-Asset Interaction** | By injecting Nifty-Gold RS and VIX Efficiency, I bridged the **0.15 correlation ceiling** for the first time. |
| **Dual-Friction Audits** | I used 0-bps vs 5-bps comparisons to isolate **Gold** as my S-Tier anchor and **Nifty** as a friction-resistant sniper. |
| **3-Fold CV Calibration** | I solved the "Honest Probability" problem, ensuring my meta-model output is a reliable predictor of true out-of-sample win-rates. |
| **Fractional Differentiation** | My manual $d$ selection (0.45) preserved ~81% of historical memory while ensuring statistical stationarity for machine learning. |

---

## 8. Finalized Labeling Approach: Triple Barrier Method
*(Calculated in `src/feature_engineering.ipynb`)*

**Source:** Marcos López de Prado, *Advances in Financial Machine Learning*

Instead of simple binary labels (up/down), a forward-scanning Triple Barrier Method is utilized. This defines three exit conditions over a rolling window $T$:
- **Upper Barrier:** Price rises by $k\sigma$ → Label = +1 (profit target hit first)
- **Lower Barrier:** Price falls by $k\sigma$ → Label = -1 (stop-loss hit first)
- **Vertical Barrier:** Time $T$ expires → Label = 0 (inconclusive / timeout)

Where $\sigma$ is the daily volatility (computed as the rolling 20-day standard deviation of daily returns).

### 8.1 Symmetric Meta-Labeling
Traditional labeling ignores the *Side* of the trade. **METR** uses a direction-aware approach:
```python
# P&L is credited based on Signal Side
Directional_Return = TB_Return * Signal
Meta_Label = 1 if Directional_Return > 0 else 0
```
This ensures that hitting a lower barrier while **Short** (-1) is correctly recorded as a "Win" (+kσ), solving the "Short Alpha Blindness" observed in early research stages.

### 8.2 Signal Conditioning (Noise Suppression)
To isolate "Trade Entry Skill" from "Market Drift," all training and evaluation is strictly filtered:
- **Condition**: Only train/evaluate on rows where `Signal != 0`.
- **Rationale**: The Meta-Filter's job is not to predict the next bar, but to judge the quality of an *active* momentum signal.

### 8.3 In-Depth Labeling Logic
Traditional binary labeling (Up/Down) forces a model to guess direction regardless of whether the move was meaningful or just noise. **METR** uses a volatility-adaptive sequence:

Entry: $P_0$ (at Open of $T+1$)  
Upper Barrier: $P_0 \cdot (1 + k\sigma)$  
Lower Barrier: $P_0 \cdot (1 - k\sigma)$  
Vertical Barrier: Time $T$ expires → Label = 0 (inconclusive / timeout)

#### Rationale for $T=5$
I found that $T=3$ was too noisy for stable directional learning. By extending to $T=5$, the **Return Spread** between profit and loss labels increases significantly across all three assets, providing a cleaner signal for the Meta-Model to filter.

#### Asset-Specific Scalings ($k$)
- **NIFTY**: $k=1.5$
- **GOLD**: $k=1.75$ (Higher volatility tail)
- **USDINR**: $k=1.5$

---

## 9. Advanced Feature Engineering: Fractional Differentiation
*(Implemented in `src/feature_engineering.ipynb`)*

To solve the stationarity-memory trade-off, **Fractional Differentiation** ($d \in [0.1, 0.9]$) was implemented. This ensures the features are stationary for ML models while retaining as much historical "memory" as possible, unlike standard integer-differencing ($d=1$).

### 9.1 Methodology
This study uses `src/frac_diff.py` with a threshold of **$10^{-4}$** to balance mathematical precision with data availability (lookback length). Instead of a strict ADF $p < 0.05$ cutoff, I selected parameters by maximizing **Memory Preservation** (correlation with original series) while achieving "good enough" stationarity.

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
- **`Vol_Ratio`**: `Vol_5d / Vol_20d`. Detects when short-term volatility is expanding relative to the monthly baseline.
- **`Vol Efficiency`**: `Ret_5d / Vol_20d`. A "Risk-Adjusted Momentum" signal identifying clean trends vs. choppy noise.
- **`ATR_Pct`**: `Average True Range / Price`. Normalizes risk across time.
- **`BB_Pct`**: Bollinger Band %B. Quantifies where price sits relative to its bands.

### 10.3 Stationarity & Memory (FracDiff)
- **`FD_Close`**: Fractionally Differentiated Close. Maintains historical memory while ensuring statistical stationarity.
- **`RSI`**: Standard Relative Strength Index (14-period).
- **`RSI_Trend`**: `RSI * Ret_5d`. A high-conviction interaction identifying "overbought but strong" vs. "overbought and weak" regimes.
- **`Gap_Intraday_Conviction`**: `Gap * Intraday_Return`. Identifies sessions where overnight sentiment and intraday action align for a powerful trend.

### 10.4 Cross-Asset Dynamics (Phase 8 Macro-Signals)
- **`Relative Strength (RS)`**: `Gold_Ret_5d - Nifty_Ret_5d`. Measures risk-appetite shifts.
- **`Usdinr_Stress_Filter`**: `Risk_Off * Nifty_Vol_Ratio`. A state-aware interaction that identifies when Indian equity stress (rising vol) aligns with a global risk-off flight. This is the project's strongest macro-filter.
- **`Cross_Vol_Ratio`**: `Nifty_Vol_20d / Gold_Vol_20d`. Measures the volatility divergence between equities and safe-havens, identifying high-regime shifts.
- **`Risk_Off`**: Binary flag; True when `Gold_Ret_5d > Nifty_Ret_5d`.
- **`Equity_Stress`**: Binary flag; True when `Nifty_Ret_5d < 0`.

### 10.5 Macro & Volatility Regime (India VIX)
- **`VIX_Relative`**: `VIX / SMA(VIX, 20)`. Identifies the current stress level relative to the monthly average.
- **`VIX_Shock`**: 1-day change in VIX. Measures the "Rate of Fear" increase.
- **`VIX_ATR_Ratio`**: `VIX / ATR_Pct`. Measures the decoupling between realized volatility (ATR) and implied volatility (VIX). High values suggest "Overpriced Fear."
- **`VIX_Momentum_Efficiency`**: `Ret_5d / (VIX_Shock + 1e-9)`. Identifies trends that are persisting *despite* rising fear.

---

## 11. Threshold Optimization (Phase 10.3)
*(Dual-Friction Audit: 0 bps vs. 5 bps)*

To bridge the gap between "Research Alpha" and "Production Alpha," I performed a forensic grid search across all thresholds $T \in [0.45, 0.60]$. The goal was to maximize **Net Calmar** (Alpha Efficiency) while monitoring **Friction Decay**.

### 11.1 The Friction Audit (Investability Lock)
I tested the strategy against a standard institutional friction of **5 basis points (bps)** per round-trip trade.

| Asset | Production Threshold ($T$) | Net Sharpe (5-bps) | Net MDD (5-bps) | Net Calmar (5-bps) | Alpha Verdict |
|-------|----------------------------|--------------------|-----------------|--------------------|---------------|
| **GOLD** | **0.52** | **1.42** | **0.18** | **7.66** | **S-Tier Anchor** |
| **NIFTY** | **0.49** | **0.63** | **0.07** | **8.71** | **Efficiency Sniper** |
| **USD/INR** | 0.50 | **-2.41** | 0.03 | < 0 | **REJECTED** |

### 11.2 Strategic Victory: The Nifty Efficiency Shift
I have identified that while Nifty achieves its "Alpha Peak" at $T=0.48$, the **"Efficiency Peak" occurs at $T=0.49$.** Moving just 0.01 in probability slashes the **Max Drawdown from 0.23 to 0.07 (a 70% reduction)**. This is my primary engineering victory for Phase 10.3.

---

## 12. Project Evolution: Phase 8 Macro-Stabilization 🏆

The project has successfully navigated the **"Overfitting Crisis"** and is now locked in its **Final Production Build**. 

- **Phase 8.5 (The Sweet Spot)**: I identified the optimal regularization grid (`max_depth: [3,4,5]`, `min_child_weight: [5,10]`) that slashed the USDINR "Memorization Gap" from **0.44 to 0.17**.
- **Phase 8.6 (Macro-Injection)**: I successfully bridged the "Interaction Wall" by injecting **Nifty Volatility** and **Gold Performance** into my USDINR meta-model.
- **Phase 8.7 (The Alpha Peak)**: I achieved a finalized production benchmark of **+10.74% Edge** on Gold and USDINR.
- **Phase 8.8 (Final Audit)**: I confirmed that **Volatility Efficiency** and the **Usdinr_Stress_Filter** are the primary drivers of my strategy outperformance.

---

## 13. Final Production Standings (Phase 10.3)

Approved metrics at the **Optimized Efficiency Thresholds**:

| Asset | Critical Threshold (T) | Test ROC AUC | Net Sharpe (5-bps) | Net Calmar | Alpha Verdict |
|---|---|---|---|---|---|
| **GOLD** | **0.52** | **0.5935** | **1.42** | **7.66** | **S-Tier Anchor** |
| **NIFTY** | **0.49** | **0.5918** | **0.63** | **8.71** | **Efficiency Sniper** |
| **USDINR** | — | 0.5489 | < 0 | < 0 | **REJECTED** |

### Strategic Verdict
I have successfully decoupled win-probability from market noise. The USDINR rejection is a **"Risk Management Premium"**—proving that institutional models must be robust to friction before being greenlit for production. My production portfolio is now limited to high-conviction, low-drawdown engines.

---

## 14. Regime Conditioning (Phase 9) — REJECTED
*(Empirical Integrity Audit)*

My attempts to integrate a **GaussianHMM** (Hidden Markov Model) to explicitly define "Crisis states" were **rejected** for inclusion in the final build. The HMM added significant complexity without outperforming the simpler, more stable **VIX_Shock** and **Nifty_Vol** indicators already present in the XGBoost architecture.

---

### 14.1 Asset-Specific Model Tuning
Each asset behaves differently and requires tailored hyperparameters:

| Asset | Recommended Approach |
|-------|---------------------|
| **Nifty (Equity)** | Trend/momentum focus, moderate depth |
| **Gold (Commodity)** | Lower learning rate (violent bursts), hedge/safety features |
| **USD/INR (FX)** | Higher regularization (central bank-managed), mean-reversion focus |

### 14.2 Regime Modeling
A volatility filter to avoid trading during high-volatility periods. This remains a potential improvement:
- I could train a separate classifier to predict High/Low volatility regimes.
- I could only take Exposure Model trades during "Calm" regimes.

### 14.3 Multi-Class & Meta-Labeling
Transitioning from binary -1, 0, +1 labeling to a model that can predict the *specific barrier hit type* with higher confidence, potentially using Meta-Labeling (as described by de Prado) to filter out false positives.

---

## 15. File Reference

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

## 16. Conclusion

The METR project successfully built a controlled experimental framework for comparing ML-based entry timing against random chance. My core finding is that **standard technical indicators provide a weak but measurably superior edge** when calibrated for symmetric alpha ($Symmetric\_AUC \approx 0.59$), but this edge is deeply sensitive to execution friction.

I identified three clear components of the production strategy:
1. **Symmetric Meta-Labeling** — to capture alpha across all price regimes.
2. **Dual-Friction Audits** — to filter out assets that memorize noise but fail under cost.
3. **Macro-Interaction Features** — to reach the institutional 0.15 correlation threshold.

My final production locks—**Gold ($T=0.52$)** and **Nifty ($T=0.49$)**—represent a mathematically robust balance of return-per-vol and tail-risk protection.

---
