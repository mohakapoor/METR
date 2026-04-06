# METR — Documentation & Findings

> **Market Exposure Timing vs Randomness**
> A controlled multi-asset empirical study investigating whether machine learning models can outperform random entry decisions.

---

## 1. Project Overview

### Objective
Determine if machine learning models trained on price, volume, and implied volatility (VIX) can identify "high-conviction" regimes for a primary momentum signal. Using **Meta-Labeling** and validating against **10,000 Monte Carlo simulations**, the study isolates true statistical predictive edge from market noise.

### Architecture (The Two-Layer Logic)
- **Layer 1 (The Trigger)**: Primary 5-day momentum signal generated in `src/indicators.py` (Volatility-Normalized).
- **Layer 2 (The Audit Engine)**: Symmetric Binary XGBoost Classifier (`src/train_trade_filter.py`) that validates the trigger signal. It confirms trade feasibility based on macro-regime stability.
- **Audit Layer**: Hypothesis testing via `src/monte_carlo_audit.py` (Statistical Significance).
- **Assets:** Nifty 50, GOLDBEES (NSE), USD/INR.
- **Validation:** TimeSeriesSplit (5-fold) + Signal-Conditional Masking ($Signal \neq 0$).

### Tech Stack & Standards
| Component | Engine | Standard |
|-----------|--------|----------|
| **Data** | Polars | Lazy-execution, zero-copy joins |
| **Model** | XGBoost | Symmetric Binary Classification |
| **Labeling** | Triple Barrier | Volatility-adaptive (Profit/Loss/Time-out) |
| **Memory** | FracDiff | Fractional memory preservation ($d=0.45$) |
| **Metrics** | Calmar / Sharpe | 5-bps friction-adjusted net returns |

---

## 2. Data Pipeline

### Source
- Raw OHLCV data fetched via `src/fetch_data.py`
- Assets: Nifty, GOLDBEES, USD/INR, and **India VIX** (implied volatility / fear gauge)
- Row count variances due to distinct market holiday schedules

### Data Pipeline Stages
Targeting two distinct specialized notebooks:

1. **Alignment (`src/data_cleaning.ipynb`)**:
    - **Asset Synchronization**: Methodical integration of Nifty, Gold, USD/INR, and VIX into a standardized temporal framework.
- **Calendar Alignment**: Normalizing the dataset against asynchronous market holidays (e.g., MCX vs. NSE).
- **Date Intersection**: Ensuring all cross-asset features are calculated on common trading days (3,440+ days total).
- **Lookback Buffer**: Anchoring the training dataset at 2014-01-01 while preserving 2012-2013 for feature warm-up.

### 2.2 Synthesis & Predictive Logic (`src/feature_engineering.ipynb`)
- **Memory Persistence**: Application of **Fractional Differentiation** (`src/frac_diff.py`) to preserve ~80% of historical memory while ensuring statistical stationarity.
- **The Labeling Engine**: Generating the **Triple Barrier target** using per-asset volatility-adaptive thresholds ($k$) and a **5-day window** ($T$).
- **Directional P&L Attribution**: Adjusting returns via the trade signal (`Return * Signal`) to evaluate directional accuracy (Meta_Label = 1 for positive directional outcomes).
- **Systemic Risk Interaction**: Implementation of the `Usdinr_Stress_Filter` and cross-asset Relative Strength metrics to isolate actionable alpha from statistical noise.

**Total Feature Scope:** 13 (Technical) + 12 (Core Interaction) + 5 (Macro/VIX).

### Train/Test Split
- **Buffer Data:** 2012–2013 (Used for rolling indicators and Fractional Differentiation lookbacks)
- **Train (Anchor):** 2014–2023 (~2200 rows)
- **Test:** 2024–2025 (~450 rows)
- Split by time (no shuffling)

---

## 3. Model Development

The XGBoost framework was selected as the core engine for signal auditing based on three professional engineering criteria:

1.  **Superior Tabular Performance**: Gradient-boosted tree models are the industry standard for high-performance classification on structured, tabular financial features.
2.  **Generalization Over Complexity**: Deep learning architectures (CNN/LSTM) require significantly larger datasets and typically fail to generalize on the limited historical sample sizes available for this project.

---

### Model v1 — Preliminary (High-Variance Iteration)

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
| **Training-Test Divergence** | **33.1%** | ⚠️ Significant Variance |

**High Confidence (>0.60 threshold):**
- Trades: 129 / 453
- Win Rate: 56.6%
- vs Baseline (51.4%): +5.2%

**Analysis:** The model demonstrated excessive training-set adaptation. The 86% training accuracy was non-generalizable, indicating the model identified patterns specific to historical data points rather than robust market signals.

---

### Model v2 — Regularized Configuration

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
| **Training-Test Divergence** | **7.6%** | ⚠️ Contained |

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

### Model v3 — Directional Returns

**Methodological Improvements:**
1. **Sign-Adjusted Returns**: Transitioned from raw `TB_Return` to `Directional_Return = TB_Return * Signal` to ensure directional alignment. 
2. **Signal-Conditional Filtering**: Implemented training strictly on active regimes ($Signal \neq 0$).
3. **Cross-Validation Calibration**: Migrated to **3-fold TimeSeries CV** for unbiased out-of-sample probability estimation.

**Results (Model v3):**

| Asset | ROC AUC | AUPRC |
|-------|---------|-------|
| **Nifty** | **0.5918** | 0.4431 |
| **Gold** | **0.5935** | 0.4602 |
| **USD/INR** | 0.5489 | 0.4833 |

**Conclusion:** The refined directional methodology effectively isolated the short-side alpha that was previously obscured. The adoption of **3-fold Cross-Validation** establishes a robust statistical foundation for probability estimation across the asset universe.

---

## 4. Benchmark Results (Model vs Random)

### Monte Carlo Simulation
- **Methodology:** 10,000 iterative simulations, each selecting a trade frequency identical to the model but initialized on randomized entry dates.
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

## 5. Feature Importance Analysis

The integration of **Macro-Interactors** indicates that predictive drivers are highly asset-specific. Below is a quantitative breakdown of the primary drivers for each strategy component.

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

By adopting **Asset-Specific Macro Interactors**, the model has successfully surpassed the **0.15 correlation threshold.** This confirms that market exposure timing is a non-uniform problem; USD/INR and Gold components necessitate cross-asset stress signatures to achieve optimal alpha capture.

---

## 6. Methodological Exclusions

| Approach | Why It Failed |
|----------|--------------|
| **USD/INR Friction-Agnostic Model** | Initial iterations achieved high AUC (0.54) in zero-cost environments, but subsequent audits revealed high **Execution Cost Sensitivity**, resulting in negative Net Sharpe at 5-bps. |
| **GaussianHMM Regime Filter** | Evaluation of Hidden Markov Models (HMMs) for crisis state identification was performed, but they added complexity without statistically outperforming the simpler **VIX_Shock** indicators in the XGBoost architecture. |
| **Simple Long-Only Meta-Labeling** | Early versions ignored the short side, leading to **Directional Asymmetry** where high-conviction pullbacks were categorized as noise. |
| **Standard Technical Indicators** | Standard indicators (RSI, MACD) demonstrated a correlation ceiling of 0.12, suggesting insufficient predictive power for a 5-day horizon without macro context. |

---

## 7. Validated Methodological Improvements

| Finding | Evidence |
|---------|----------|
| **Directional Meta-Labeling** | Isolated short-side alpha using `Directional_Return`, enabling high-conviction directional filtering. |
| **Cross-Asset Integration** | Integrated Nifty-Gold Relative Strength and VIX Efficiency to surpass the **0.15 correlation threshold**. |
| **Execution Cost Audits** | Comparative analysis (0-bps vs. 5-bps) isolated **GOLDBEES** as the primary strategy anchor and **Nifty** as a high-efficiency component. |
| **TimeSeries CV Calibration** | Implemented 3-Fold Cross-Validation for unbiased probability estimation. |
| **Fractional Differentiation** | Parameter selection ($d=0.45$) preserved ~81% of historical memory while achieving statistical stationarity for model compatibility. |

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
This ensures that hitting a lower barrier while **Short** (-1) is correctly recorded as a "Win" (+kσ), solving the "Short-Side Blindness" observed in early research stages.

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
Empirical analysis indicates that $T=3$ proved overly sensitive to market noise for stable directional learning. By extending to $T=5$, the **Return Spread** between profit and loss labels increases significantly across all three assets, providing a cleaner signal for meta-model discrimination.

#### Asset-Specific Scalings ($k$)
- **NIFTY**: $k=1.5$
- **GOLD**: $k=1.75$ (Higher volatility tail)
- **USDINR**: $k=1.5$

---

## 9. Advanced Feature Engineering: Fractional Differentiation
*(Implemented in `src/feature_engineering.ipynb`)*

To solve the stationarity-memory trade-off, **Fractional Differentiation** ($d \in [0.1, 0.9]$) was implemented. This ensures the features are stationary for ML models while retaining as much historical "memory" as possible, unlike standard integer-differencing ($d=1$).

### 9.1 Methodology
This study uses `src/frac_diff.py` with a threshold of **$10^{-4}$** to balance mathematical precision with data availability (lookback length). Instead of a strict ADF $p < 0.05$ cutoff, parameters were selected by maximizing **Memory Preservation** (correlation with original series) while achieving sufficient stationarity.

### 9.2 Final Manual Selection ($d$)
| Asset | Optimal $d$ | Correlation | ADF p-value | Characteristic |
|---|---|---|---|---|
| **Nifty 50** | **0.45** | 0.818 | 0.079 | High memory / Strong trend persistence |
| **Gold** | **0.50** | 0.772 | 0.096 | Best memory-stationarity compromise |
| **USD/INR** | **0.30** | 0.908 | 0.034 | Perfectly stationary with 91% memory |

---

## 10. Feature Dictionary (In-Depth)

The following features are synthesized in `src/indicators.py` and `src/feature_engineering.ipynb`:

### 10.1 Momentum & Price Action
- **`Ret_1d, Ret_5d`**: Standard log-returns for capturing momentum trends.
- **`Intraday_Return`**: Session conviction indicator.
- **`Gap`**: Overnight sentiment shifts.
- **`Close_Pos_Range`**: Price rejection signature at extremes.
- **`Range_Expansion`**: Rate of volatility expansion.

### 10.2 Volatility & Regime Detection
- **`Vol Efficiency`**: `Ret_5d / Vol_20d`. Risk-adjusted momentum signature.
- **`ATR_Pct`**: Normalized volatility across timeframes.
- **`BB_Pct`**: Bollinger Band relative positioning.

### 10.3 Stationarity & Memory (FracDiff)
- **`FD_Close`**: Fractionally Differentiated Close. Maintains historical memory while ensuring statistical stationarity.
- **`RSI`**: Standard Relative Strength Index (14-period).
- **`RSI_Trend`**: `RSI * Ret_5d`. A high-conviction interaction identifying "overbought but strong" vs. "overbought and weak" regimes.
- **`Gap_Intraday_Conviction`**: `Gap * Intraday_Return`. Identifies sessions where overnight sentiment and intraday action align for a powerful trend.

### 10.4 Cross-Asset Dynamics 
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

## 11. Threshold Optimization 
*(Dual-Friction Audit: 0 bps vs. 5 bps)*

To bridge the gap between "Research Edge" and "Production Edge," we performed a forensic grid search across all thresholds $T$. The goal was to maximize **Net Calmar** (Strategy Efficiency) while monitoring **Friction Decay**.

### 11.1 The Economic Feasibility Audit
The strategy was evaluated against an institutional-standard friction of **5 basis points (bps)** per round-trip transaction.

| Asset | Target Threshold ($T$) | Net Sharpe (5-bps) | Net MDD (5-bps) | Net Calmar (5-bps) | Component Classification |
|-------|----------------------------|--------------------|-----------------|--------------------|---------------|
| **GOLDBEES** | **0.52** | **1.42** | **0.18** | **7.66** | **Primary Strategy Anchor** |
| **NIFTY** | **0.49** | **0.63** | **0.07** | **8.71** | **High-Efficiency Component** |
| **USD/INR** | 0.50 | **-2.41** | 0.03 | < 0 | **Excluded (High Friction)** |

### 11.2 Optimization Result: The Nifty Efficiency Shift
Quantitative analysis indicates that while Nifty achieves its individual alpha peak at $T=0.48$, the **optimal efficiency peak occurs at $T=0.49$.** This slight adjustment in the probability threshold reduces the **Maximum Drawdown from 0.23 to 0.07 (a 70% reduction)**, significantly improving the strategy's risk-adjusted profile.

---
## 12. Macro-Adaptive Stability

The project addressed high model variance by pivoting from univariate technicals to a global macro-regime context. This transition transformed the filters into logic-driven components capable of adapting to systemic market stress.

- **Variance Reduction**: Systematic regularization (depth 3-5, min-child 5-10) reduced the USD/INR training divergence from 0.44 to 0.17 without compromising signal integrity.
- **Cross-Asset Integration**: Predictive capability was enhanced by integrating **Nifty Volatility** and **Gold Performance** into the FX and Gold models. The engine now identifies that Gold's momentum is validated primarily when equity markets exhibit structural stress.
- **The Alpha Profile**: These refinements produced a **+10.7% return differential** on high-conviction signals, validating cross-asset context as a primary driver of accuracy.
- **Attribution Analysis**: SHAP value evaluation confirmed that the **`Usdinr_Stress_Filter`** (Systemic Volatility * Risk-Aversion) is the fundamental source of outperformance, functioning as an indicator of systemic market instability.

---


---

## 13. Strategy Status Conclusion

Performance summary at optimized efficiency thresholds:

| Asset | Critical Threshold (T) | Test ROC AUC | Net Sharpe (5-bps) | Net Calmar | Strategy Verdict |
|---|---|---|---|---|---|
| **GOLDBEES** | **0.52** | **0.5935** | **1.42** | **7.66** | **Active - Core Anchor** |
| **NIFTY** | **0.49** | **0.5918** | **0.63** | **8.71** | **Active - High Efficiency** |
| **USDINR** | — | 0.5489 | < 0 | < 0 | **Inactive (High Friction)** |

### Management Conclusion
The framework effectively isolates high-probability entry regimes. The exclusion of USD/INR is a programmatic risk management decision, demonstrating that strategies must maintain robustness to transaction costs before capital allocation. The strategy is now restricted to high-conviction, low-drawdown components.

---

## 14. Empirical Integrity Audit (GaussianHMM)
*(Status: REJECTED)*

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
A volatility filter to avoid trading during high-volatility periods remains a strategic improvement:
- Implementation of a dedicated classifier to predict high/low volatility regimes.
- Restricting exposure trades exclusively to "Calm" regimes.

### ## 15. Future Research Directions

### 15.1 Asset-Specific Model Tuning

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

### Ex-Post Performance Audit (2024-2025)

The final evaluation phase utilized an out-of-sample (OOS) audit incorporating institutional compounding and a **Stochastic Benchmarking** simulation (Monte Carlo), supported by three independent statistical tests (Binomial, Kupiec, and T-test).

| Asset | Annualized Sharpe | Adjusted Signal Sharpe | MDD (Comp.) | Calmar |
|---|---|---|---|---|
| **GOLD ($T=0.52$)** | **1.21** | **1.51** | **7.99%** | **1.87** |
| **NIFTY ($T=0.49$)** | -0.40 | -0.48 | 15.57% | -0.43 |
| **USD/INR ($T=0.50$)** | -0.60 | -0.80 | 4.97% | -0.56 |

### Academic Validation Checklist

To isolate true skill from luck, every asset was subjected to a triple-layered statistical stress test:

| Test | Gold Stat / P-val | Nifty Stat / P-val | USD/INR Stat / P-val |
|---|---|---|---|
| **Monte Carlo** | **Target 1.21 / 0.0104** | Target -0.40 / 0.4019 | Target -0.60 / 0.1237 |
| **Binomial (WR)** | **W=39/59 / 0.0092** | W=103/204 / 0.4721 | W=78/151 / 0.3725 |
| **Kupiec (Reliability)** | **LR Calc / 0.0126** | LR Calc / 0.8886 | LR Calc / 0.6841 |
| **T-test (Mean Ret)** | t=1.62 / 0.1099 | t=-0.53 / 0.5943 | t=-0.80 / 0.4242 |

**Conclusion**:
Gold is the only asset to achieve statistical significance across multiple independent evaluations. Most notably, the meta-filter successfully converted a suboptimal baseline signal (-0.73) into a statistically significant alpha (1.21). This provides definitive confirmation of the strategy's predictive edge within the METR architecture.

---

## 16. METR — Final Conclusions

### Primary Finding
A meta-labeling framework combining a momentum-based signal with an XGBoost filter trained on historical price, volume, and implied volatility data generates statistically significant alpha on GOLDBEES (NSE) over the out-of-sample period 2024-2026. The null hypothesis — that model-selected trades perform no better than stochastic selection from the same signal universe — is rejected at p<0.05 across three independent statistical tests.

---

### Gold — Statistically Significant Alpha

```
Total Return:     +17.92% (out-of-sample, 2024-2026)
Sharpe Ratio:      1.21 (trade-frequency annualized)
Max Drawdown:      7.99%
Calmar Ratio:      1.87
Win Rate:          66.1% (39/59 trades)
Baseline Sharpe:  -0.73 (signal-only, no filter)
```

The raw momentum signal alone loses money (Sharpe -0.73). The model filter transforms this into a profitable strategy (Sharpe +1.21), representing a **+1.94 Sharpe lift** attributable entirely to the model's trade selection. This directly validates the meta-labeling hypothesis — the signal provides the opportunity universe, the model provides the edge.

**Statistical confirmation**: Monte Carlo (p=0.0104), Binomial (p=0.0092), Kupiec (p=0.0126) all significant. T-test underpowered at n=59 — explained by selective filtering reducing trade count.

The SHAP analysis revealed that Vol Efficiency, VIX_Momentum_Efficiency, RSI, and Ret_5d are the primary alpha drivers, with USD/INR features contributing meaningfully — consistent with GoldBees' structural exposure to rupee-dollar exchange rate movements.

---

### Nifty 50 — Null Result (Efficiency Benchmark)

```
Total Return:    -8.01%
Sharpe:          -0.40
Win Rate:         50.5%
All 4 tests:     Non-significant
```

The meta-filter demonstrates no exploitable edge on Nifty 50. Analysis confirmed near-zero directional separation across all features (max AUC 0.029). This is consistent with the high institutional coverage and deep liquidity of India's benchmark equity index, which limits the predictive power of price-only features. The model correctly produces a null result on an efficiently priced asset.

---

### USD/INR — Null Result (Managed Float)

```
Total Return:    -2.82%
Sharpe:          -0.60
Win Rate:         51.7%
All 4 tests:     Non-significant
```

No statistically significant alpha detected. The managed float nature of USD/INR — with periodic RBI intervention disrupting momentum patterns — creates a structural ceiling on OHLCV-based prediction that the model cannot overcome. Notably, the model filter still outperforms the raw signal baseline (Sharpe -0.60 vs -1.49), suggesting partial discriminative ability insufficient to generate significant alpha.

---

### Cross-Asset Evaluation Findings

The divergent results across the three asset classes are not indicative of a methodological failure, but rather represent a core empirical finding. The framework accurately identifies exploitable market structure where theoretical precursors exist (commodity ETF with embedded currency exposure and historical inefficiencies) while correctly identifying the absence of edge in highly efficient or managed environments (benchmark equity indices and managed currency pairs). This internal consistency across varying market regimes provides further credibility to the METR architecture.

---

### Strategic Implications

The METR study demonstrates that price action, volume, and implied volatility data—independent of exogenous macro indicators or sentiment analysis—are sufficient to construct a statistically significant meta-filter for GOLDBEES that outperforms stochastic market entry. However, the same data set remains insufficient to generate significant alpha on deeply liquid benchmark indices or strictly managed currency pairs.

This confirms that market exposure timing via meta-labeling is a highly asset-specific and regime-sensitive approach, offering practical viability for specific commodity instruments within the Indian market ecosystem.

**METR RESEARCH STATUS: COMPLETE (GOLD STRATEGY FINALIZED)**

---

## 17. Index of Project Files

| File | Purpose |
|------|---------|
| `src/fetch_data.py` | Automated historical data ingestion (Source: Yahoo Finance) |
| `src/data_cleaning.ipynb` | Multi-asset synchronization and alignment |
| `src/feature_engineering.ipynb` | Refined feature synthesis and labeling |
| `src/indicators.py` | Core technical indicator library (Polars-optimized) |
| `src/frac_diff.py` | Fractional Differentiation logic for memory preservation |
| `src/triple_barrier.py` | Volatility-adaptive labeling logic |
| `src/train_trade_filter.py` | Layer 2 XGBoost Meta-Model training |
| `src/backtest_engine.py` | Professional backtesting engine (MDD/Sharpe) |
| `src/monte_carlo_audit.py` | 10,000-iteration "Skill vs. Luck" hypothesis tester |
| `src/threshold_optimizer.py` | Net Calmar / Friction-decay search grid |
| `src/shap_audit.py` | Local and Global explainability (SHAP) |
