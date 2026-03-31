# METR — Lab Logs & Observations

This file tracks daily experiments, grid searches, and key insights over the course of the METR project.

---

### 2026-03-18 — Triple Barrier Labeling: Grid Analysis (Nifty)

**What was done:**
Implemented `src/tripple_barrier.py` with a forward-scan labeler. Ran a grid search over `k ∈ [0.5, 0.75, 1.0, 1.5]` × `T ∈ [3, 5, 10]` on Nifty training data (~2447 rows).

**Key Results:**

| k | T | -1 (%) | 0 (%) | +1 (%) | Notes |
|---|---|--------|-------|--------|-------|
| 0.5 | 3 | 63.6 | 0.2 | 36.2 | Barriers too tight — nearly all hit same day |
| 0.75 | 3 | 55.1 | 3.5 | 41.4 | Still tight, heavy -1 skew |
| 1.0 | 3 | 47.9 | 10.4 | 41.6 | Moderate — meaningful timeout class appears |
| 1.5 | 3 | 35.6 | 33.9 | 30.5 | Most balanced 3-class distribution |
| 1.0 | 5 | 51.4 | 2.7 | 45.9 | Longer T reduces timeouts |
| 1.5 | 5 | 43.2 | 14.9 | 41.9 | Good balance with more time |

**Observations:**
1. **Persistent -1 bias across all configurations.** Two causes identified:
   - **Same-day tiebreak:** When both barriers are hit in one day (common at low k), the code defaults to `-1`. At `k=0.5`, this dominates the label.
   - **Market microstructure:** Intraday lows tend to be further from Open than Highs (negative skew), so the lower barrier gets hit first naturally.
2. **k=0.5 and k=0.75 are too tight** — barriers fall inside a single day's range, so the label mostly measures intraday skew, not directional signal.
3. **Increasing T beyond 5 adds almost nothing** — most barriers are hit within the first few days regardless.
4. **Best candidates:** `k=1.0, T=3` (if you want some timeout filtering) or `k=1.5, T=3` (if you want balanced classes).

---

### 2026-03-19 — Triple Barrier Labeling: Multi-Asset Grid Analysis

**What was done:**
Refined `src/tripple_barrier.py` to use a symmetric barrier ($k$) and evaluate over a wider grid ($k \in [0.5, 0.75, ..., 2.5, 3.0]$ and $T \in [3, 5, 10]$).
Replaced feature-label correlation with a **Balance Score** metric to evaluate configurations, given the weak predictive power of the current features.
- **Balance Score** measures the sum of squared deviations from a perfectly balanced distribution (33.3% per class). Lower is better.
- Applied a **hard filter**: configurations where any single class exceeds 45% are rejected to prevent the model from learning a strong majority-class bias.

**Key Results (Mathematical Optimal vs Practical Chosen):**

*1. Mathematically Optimal (pure balance)*
| Asset | k | T | Balance Score | -1 (%) | 0 (%) | +1 (%) |
|---|---|---|---|--------|-------|--------|
| **Nifty** | 2.0 | 5 | 0.4 | 33.7% | 33.5% | 32.8% |
| **Gold** | 2.5 | 5 | 15.4 | 34.8% | 35.0% | 30.1% |
| **USDINR** | 2.0 | 5 | 1.6 | 32.3% | 34.0% | 33.7% |

*2. Practically Chosen (realistic market moves)*
| Asset | k | T | -1 (%) | 0 (%) | +1 (%) |
|---|---|---|--------|-------|--------|
| **Nifty** | 1.5 | 3 | 35.6% | 33.9% | 30.5% |
| **Gold** | 1.75 | 3 | 41.6% | 31.2% | 27.3% |
| **USDINR** | 1.5 | 3 | 34.7% | 31.2% | 34.1% |

**Observations & Final Logic:**
1. **Mathematical Optimization Overfits to Outliers:** While the algorithm found near-perfect mathematical balance at $T=5$ and $k \in \{2.0, 2.5\}$, expecting an asset to move $2.5\sigma$ in just 5 days is statistically highly improbable. Using these parameters would force the XGBoost model to hunt for extreme outlier events rather than tradeable signals.
2. **The $T=3$ Practical Choice:** A shorter time horizon ($T=3$) with tighter, more realistic barriers ($k=1.5$ to $k=1.75$) yields a slightly less "perfect" mathematical score but a vastly superior practical trading goal. It still maintains a healthy ~30-34% timeout class, proving the noise filter is functioning.
3. **Asset Volatility Profile:** Gold requires slightly wider barriers ($k=1.75$) than Nifty and USDINR ($k=1.5$) to achieve this balance, reflecting its higher relative intra-period volatility (longer tails).

**Next Steps:**
- Add `TB_Label` to the data pipeline (`src/data_cleaning.ipynb`), computed using the identified best parameters per asset.
- Retrain XGBoost as a 3-class classifier (`multi:softprob`).

---

### 2026-03-20 — Triple Barrier Labeling: Edge Case Tiebreaker Update

**What was done:**
Updated the tie-breaker logic in `generate_barriers()`. Previously, if both the upper and lower barriers were hit on the exact same day, the algorithm conservatively defaulted to `-1` (assuming the stop-loss hit first). This has been changed to default to `0` (Timeout / Neutral).

**Key Observations from the Logs:**
1. **Elimination of Artificial '-1' Skew:** The previous logic caused a massive artificial inflation of the `-1` class, especially when using tighter barriers where hitting both limits intraday is common. For example, on USDINR at `k=0.5, T=3`, the `-1` class plummeted from **64.9%** down to **31.5%** after the fix.
2. **Choppy Assets Benefited Most:** USDINR inherently has high intraday chop relative to its directional trends. When both limits are breached intraday, labeling it a loss (`-1`) penalized the model. Labeling it `0` correctly identifies that this specific timeframe was directionless/choppy volatility rather than a clean directional loss.
3. **Balanced Distributions:** As a result of this change, the overall label distributions are significantly more natural and balanced across all three assets without mathematically forcing it.

**Final Best Parameters & Evaluation Metric Switch:**
I formally selected the final parameters by evaluating them against the **Return Spread** (+1 Mean Return minus -1 Mean Return) rather than purely optimizing the mathematical **Balance Score** (Imbalance). 

*Why Spread > Imbalance:*
Mathematical balancing algorithms blindly force the distribution toward 33.3% per class. As seen previously on Gold, this forced the model onto extreme statistical outliers ($2.5\sigma$ in 5 days) just to satisfy the equation, starving the model of normal trade setups. **Spread** is vastly superior because it measures the actual *financial edge* of the label. When a $+1$ is triggered, I need proof that the asset drifted correctly, and similarly that $-1$ actually captured a loss. Maximizing this spread ensures the label accurately isolates true directional momentum/alpha, rather than just mathematically grouping noise perfectly into thirds.

*Final Parameter Selection (Fixed at $T=3$):*

| Asset | $k$ | $T$ | -1 (%) | 0 (%) | +1 (%) | Return Spread |
|---|---|---|--------|-------|--------|--------|
| **Nifty 50** | 1.5 | 3 | 35.2% | 34.3% | 30.5% | 2.72% |
| **Gold** | 1.75 | 3 | 38.7% | 34.1% | 27.3% | 2.71% |
| **USD/INR** | 1.5 | 3 | 33.3% | 32.7% | 34.1% | 0.82% |

These practical boundaries successfully pass the balance filter (no class > 45%), provide a very healthy timeout rate (~34% noise removed), and maintain strong directional spreads.

---

### 2026-03-27 — Fractional Differentiation: Stationarity vs. Memory

**What was done:**
Implemented `src/frac_diff.py` to find the optimal differentiation order $d$ that achieves stationarity (ADF p-value < 0.05) while preserving maximum memory. Tested $d \in [0.1, 0.9]$ at threshold 1e-5.

**Key Results:**

| Asset | Optimal d | ADF p-value | Plot Reference |
|---|---|---|---|
| **Nifty 50** | 0.40 | 0.0072 | `nifty_d_value_vs_p_value.png` |
| **Gold** | 0.30 | 0.0214 | `gold_d_value_vs_p_value.png` |
| **USD/INR** | 0.30 | 8.14e-07 | `usdinr_d_value_vs_p_value.png` |

**Observations:**
1. **USD/INR is hyper-stationary:** Even at $d=0.3$, the p-value is extremely low ($10^{-7}$). This suggests the raw series has very little long-term memory or is heavily mean-reverting.
2. **Nifty requires more differencing:** Nifty did not pass the stationarity test until $d=0.4$, confirming it has stronger trend persistence than Gold or USD/INR.
3. **Optimal Mapping:** These $d$ values are saved to `config.yaml` to prevent over-differencing (memory loss) in the next phase.

**Next Steps:**
- Integrate these $d$ values into the feature pipeline in `src/data_cleaning.ipynb`.
- Align cross-asset timestamps for unified modeling.

---

### 2026-03-29 — Fractional Differentiation: Threshold Tuning & Data Expansion

**What was done:**
Refined the fractional differentiation pipeline by testing different `thresh` values ($10^{-3}, 10^{-4}, 10^{-5}$) to balance the trade-off between mathematical precision and practical data availability (Lookback Length $L$).

**Lookback Analysis (Threshold $10^{-5}$):**
At the previous threshold of $10^{-5}$, small $d$ values required excessive historical data:
- $d=0.30 \rightarrow L=2275$ days (~9 years of lookback)
- $d=0.10 \rightarrow L=4076$ days (~16 years of lookback)

Using $10^{-5}$ would cause too many `NaN` values, effectively starving the model of recent training data.

**Key Findings & Adjustments:**
1. **Threshold Compromise ($10^{-4}$):** Found $10^{-4}$ to be the optimal compromise. It significantly reduces the lookback period compared to $10^{-5}$ while maintaining much higher feature stability than the aggressive $10^{-3}$ threshold (which forced $d$ values too high).
2. **Data Expansion (2012-2013):** Updated `src/fetch_data.py` to include two extra years of data (2012 and 2013). This provides a necessary buffer for rolling features and fractional differentiation.
3. **New Training Anchor:** Formally set the training start date to **2014-01-01**. This ensures that by the first training row, all features (including those with long memory) are fully populated with real data.

**Updated $d$-Values (at Threshold $10^{-4}$):**
| Asset | Optimal d | ADF p-value |
|---|---|---|
| **Nifty 50** | 0.50 | 0.041 |
| **Gold** | 0.60 | 0.027 |
| **USD/INR** | 0.30 | 8.14e-07 |

### 2026-03-30 — Fractional Differentiation: Manual Parameter Selection

**What was done:**
Refined the final differentiation order ($d$) by manually evaluating the trade-off between statistical stationarity (ADF p-value) and memory preservation (Correlation with original series).

**Manual Selections & Trade-offs:**

Instead of relying on a strict p-value threshold ($<0.05$), the following manual choices were made to prioritize higher memory preservation for the Nifty and Gold models:

| Asset | Chosen d | Correlation | ADF p-value | Rationale |
|---|---|---|---|---|
| **NIFTY** | **0.45** | 0.818 | 0.079 | Best balance; $d=0.50$ dropped correlation too significantly (to 0.76). |
| **GOLD** | **0.50** | 0.772 | 0.096 | Optimal compromise; $d=0.55+$ aggressively degrades memory. |
| **USDINR** | **0.30** | 0.908 | 0.034 | Passes stationarity threshold with very high memory preservation. |


**Final Decision:**
These manual overrides ensure the features maintain a strong long-term memory (0.77 to 0.91 correlation), which is critical for the predictive model, even if Nifty and Gold are slightly below the strict 95% confidence interval for stationarity.

**Next Steps:**
- Integrate these finalized $d$ values into the feature pipeline.
- Proceed with cross-asset data alignment anchoring at 2014-01-01.

---

### 2026-03-31 — Phase 4: Multi-Class XGBoost Training & Feature Analysis

**What was done:**
Upgraded the training pipeline (`src/train_exposure.py`) from binary classification to multi-class (`multi:softprob`, 3 classes: -1, 0, +1 remapped to 0, 1, 2). Unified mode only — each asset trains on `Exposure_Features + Cross_Asset_Features[asset]`. Ran full GridSearchCV (128 combinations × 5 folds) on an A5000 GPU. Separately ran `src/feature_analysis.py` (updated for one-vs-rest correlations) across all 3 assets.

---

#### Training Results — Nifty Unified (28 features)

| Metric | Value |
|---|---|
| Best CV Accuracy | 0.4123 |
| Train Accuracy | 0.5768 |
| Test Accuracy | 0.3942 |
| Overfit Gap | 0.1826 ⚠️ |
| High-Conf +1 trades (>0.60) | None |

**Best Params:** `max_depth=3, lr=0.03, n_estimators=100, colsample_bytree=0.9, subsample=0.7, reg_alpha=0.1, reg_lambda=1.0`

| Class | Precision | Recall | F1 |
|---|---|---|---|
| -1 (stop-loss) | 0.35 | 0.54 | 0.42 |
| 0 (timeout) | 0.51 | 0.47 | 0.49 |
| +1 (profit) | 0.30 | **0.12** | 0.17 |

#### Training Results — Gold Unified (27 features)

| Metric | Value |
|---|---|
| Best CV Accuracy | 0.4137 |
| Train Accuracy | 0.6806 |
| Test Accuracy | 0.4744 |
| Overfit Gap | 0.2062 ⚠️ |
| High-Conf +1 trades (>0.60) | None |

**Best Params:** `max_depth=4, lr=0.05, n_estimators=100, colsample_bytree=0.9, subsample=0.7, reg_alpha=0.1, reg_lambda=0.5`

| Class | Precision | Recall | F1 |
|---|---|---|---|
| -1 (stop-loss) | 0.48 | 0.71 | 0.58 |
| 0 (timeout) | 0.47 | 0.62 | 0.53 |
| +1 (profit) | 0.38 | **0.04** | 0.07 |

#### Training Results — USDINR Unified (26 features)

| Metric | Value |
|---|---|
| Best CV Accuracy | 0.3735 |
| Train Accuracy | 0.5605 |
| Test Accuracy | 0.3675 |
| Overfit Gap | 0.1930 ⚠️ |
| High-Conf +1 trades (>0.60) | None |

**Best Params:** `max_depth=4, lr=0.05, n_estimators=100, colsample_bytree=0.9, subsample=0.7, reg_alpha=0.0, reg_lambda=1.0`

| Class | Precision | Recall | F1 |
|---|---|---|---|
| -1 (stop-loss) | 0.29 | 0.29 | 0.29 |
| 0 (timeout) | 0.36 | 0.23 | 0.28 |
| +1 (profit) | 0.42 | **0.56** | 0.48 |

---

#### Key Findings from Feature Analysis & Results

**1. The Timeout Magnet — Vol_20d**
`Vol_20d` showed the single strongest correlation with label `0` (timeout) across all assets:
- Nifty: `+0.3947` vs timeout
- Gold: `+0.3857` vs timeout

In high-volatility regimes the 3-day barriers are rarely breached cleanly — price oscillates within the bands returning label 0. This means `Vol_20d` dominates model decisions toward timeout prediction, suppressing +1/-1 recall. It functions as a regime detector but at the cost of directional signal.

**2. Divergent +1 Recall (Gold vs USDINR)**
- **Gold (0.04 recall)**: The model is almost entirely blind to profit setups in Gold, defaulting heavily to predicting stop-losses. This suggests the current features for Gold are primarily "fear indicators" (matching its -1 correlate).
- **USDINR (0.56 recall)**: Surprisingly, USDINR shows the best directional capture for the +1 class. While absolute accuracy is lower (0.37), the model has a clear preference for predicting profit setups, likely capturing the slow-drift nature of the currency.

**3. +1 Recall Crisis in Equities (0.12 for Nifty)**
The model catches only 12% of actual profitable moves. The strongest +1 predictors found were cross-asset features:
- `Gold_Nifty_RS_5d`: +0.1367 vs +1
- `Risk_Off`: +0.1171 vs +1

**4. Momentum Dynamics**
- Nifty: `ROC_10 → -0.1727 vs +1`. Strong momentum precedes reversals.
- Gold: `MACD_Hist → +0.1286 vs +1`. Momentum predicts continuation.

---

#### Feature Reduction Decision

Several features found to be redundant or harmful and will be dropped before next training run:

| Dropped Feature | Reason |
|---|---|
| `Ret_1d`, `Ret_3d` | Subsumed by `Ret_5d`; lower gain, same directional signal |
| `Trend_Strength` | Mathematically identical to `Ret_20d` |
| `Vol_20d` | Timeout-magnet; absolute vol already captured by `ATR_Pct` |
| `FD_Close_Lag1` | 1-day shift of `FD_Close` with minimal incremental signal |
| `RSI` | Weaker than `ROC_10` for both Nifty and Gold |
| `MA_Ratio` | Weaker than `Price_vs_MA20` |
| `Close_Pos_Range` | Signal already captured by `Intraday_Return` and `BB_Pct` |

**Result: 21 → 13 Exposure Features.** Config updated accordingly.

---

### 2026-03-31 — Phase 4: Iteration 2 (Trimmed Feature Set)

**What was done:**
Retrained all 3 unified models using the trimmed 13 per-asset technical features (removed 8 redundant/noise features). Compared metrics against the Iteration 1 baseline.

---

#### Comparison Summary (Iteration 1 vs. Iteration 2)

| Asset | Test Accuracy ( $\Delta$ ) | +1 Recall ( $\Delta$ ) | Overfit Gap ( $\Delta$ ) | Target Status |
|---|---|---|---|---|
| **NIFTY** | **0.4209 (+2.6%)** | 0.11 (-1.0%) | 0.1853 (Same) | **IMPROVED ACC** |
| **GOLD** | 0.4499 (-2.4%) | 0.04 (Same) | 0.2339 (+2.7%) | **DECLINED** |
| **USDINR** | 0.3742 (Same) | 0.42 (-14.0%) | 0.1875 (Same) | **LOST SIGNAL** |

---

#### Detailed Results — Nifty (Balanced 13 features)

| Metric | Value | Baseline Comparison |
|---|---|---|
| Test Accuracy | 0.4209 | +2.6% |
| -1 Recall (SL) | **0.69** | +15.0% |
| 0 Recall (TO) | 0.41 | -6.0% (Better) |
| +1 Recall (TP) | 0.11 | Stagnant |

**Observation:** Removing `Vol_20d` and other redundant features significantly helped Nifty's overall accuracy and its ability to identify -1 signals. The model is now less of a "timeout magnet," but still fails to pick up the +1 directional signal.

#### Detailed Results — USDINR (Balanced 13 features)

| Metric | Value | Baseline Comparison |
|---|---|---|
| Test Accuracy | 0.3742 | Same |
| +1 Recall (TP) | 0.42 | -14.0% |

**Observation:** Trimming features hurt USDINR's directional prediction capacity. The "management" of the currency likely requires a broader feature set to capture subtle deviations, even if they appear collinear on the surface.

---

#### Strategic Pivot: Forward-Looking Volatility

The "Profit Hunter" problem (low +1 recall) persists because our features are lagging technicals. 
- **Absolute Realized Volatility** (Vol_20d) is a noise magnet.
- **Relative Realized Volatility** (Vol_Ratio) helps, but isn't enough.

**Next Priority:** **India VIX Integration**.
VIX provides a forward-looking expectation of volatility. This is expected to:
1. Improve the **Triple Barrier width** (making it regime-aware).
2. Clean the **Feature Space** (replacing lagged vol with forward expectations).
3. Increase **+1 Recall** by segregating noise from true volatility expansion.

---

## **Phase 4: Final Experimental Conclusion**

Based on empirical evaluation of triple barrier labeling across multiple horizons and barrier widths, **T = 5** was selected as the optimal prediction horizon.

This choice reflects the best trade-off between:
* **Label clarity (reduced noise)**
* **Class balance (usable distribution of -1 / 0 / +1)**
* **Economic significance (higher return spread)**

---

### **Why T = 5 is Optimal**

#### 1. T = 3 → Too Noisy
At T = 3:
* Labels are dominated by **short-term randomness**
* Higher proportion of **ambiguous or weak outcomes**
* Lower return separation between +1 and -1

**Result:** The model fails to learn meaningful directional patterns and defaults toward neutral or defensive predictions.

#### 2. T = 5 → Best Balance
At T = 5:
* **Return spread increases** across all assets
* **+1 labels become more meaningful** (stronger average returns)
* **Class distribution stabilizes** (reduced dominance of any single class)
* Noise reduces while still preserving enough samples

**Result:** Labels become both **learnable and economically relevant**, enabling the model to capture short-term directional structure.

#### 3. T = 7 and T = 10 → Diminishing Returns
At higher horizons:
* Marginal improvement in spread
* Increasing dominance of **strong, obvious moves only**
* Reduction in **sample diversity and learnability**
* Shift away from short-term dynamics toward medium-term trends

**Result:** Labels become cleaner but **less representative of typical market behavior**, reducing generalization and weakening model robustness.

---

### **Asset-Specific k Values**

Different assets exhibit different volatility structures, so barrier widths must be calibrated accordingly.

#### **NIFTY → k = 1.5**
* Produces **strong return separation (~2%+)**
* Minimizes ambiguous (0) labels
* Maintains **balanced directional classes (~45–50%)**

→ Captures meaningful index-level moves without excessive noise.

#### **GOLD → k = 1.75**
* Gold requires **wider barriers** due to higher volatility persistence
* k = 1.75:
  * Reduces noise from small fluctuations
  * Increases **directional clarity and payoff magnitude**
* Avoids over-filtering seen at extreme k (e.g., 2.0)

→ Focuses on **high-confidence macro-driven moves**.

#### **USDINR → k = 1.5**
* FX exhibits **lower volatility and tighter ranges**
* k = 1.5:
  * Reduces neutral outcomes
  * Maintains sufficient +1 / -1 balance
  * Improves spread (~1%) without over-filtering

→ Provides **usable directional signal in a low-volatility regime**.

---

### **Final Justification**

The selected configuration:

```yaml
Triple_Barrier:
  T: 5
  k_nifty: 1.5
  k_gold: 1.75
  k_usdinr: 1.5
```

represents an optimal balance between **signal strength (return spread)**, **label quality (reduced noise)**, and **model learnability (sufficient sample diversity)**. 

**Bottom Line:** T = 5 was selected because it is the first horizon where market structure becomes statistically learnable without drifting into slower, less relevant dynamics.

