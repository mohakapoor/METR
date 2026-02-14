# METR — Market Exposure Timing vs Randomness
> **A Controlled Multi-Asset Empirical Study**

## 1. Executive Summary
The **METR** project investigates whether structured machine learning models can outperform pure randomness in short-term exposure decisions. By using a fixed 3-day holding period and comparing model decisions against a 50/50 "Long vs. Flat" random baseline, the study isolates the statistical "edge" provided by market-derived features.

## 2. Research Objective
The goal is to determine if historical price dynamics (momentum, mean reversion, and volatility regimes) contain predictive power that exceeds a random coin-flip over an intermediate horizon.

**Key Experimental Constants:**
* **Holding Period:** Fixed at 3 Days (to balance signal decay vs. market noise).
* **Baseline:** Monte Carlo simulations using a 50/50 random generator (Long or Flat).
* **Architecture:** Localized modeling (specific models for each asset class).



---

## 3. Asset Universe & Model Structure
The study employs a **Triple-Asset, Local-Model** architecture. Each asset has its own dedicated Exposure and Regime models to account for unique volatility profiles.

| Asset | Model Pair | Logic |
| :--- | :--- | :--- |
| **Equity (e.g., SPY/NIFTY)** | Exposure + Regime | Growth/Risk-On dynamics. |
| **Gold (e.g., GLD/GoldBeES)** | Exposure + Regime | Safe-haven/Inflation dynamics. |
| **Currency (e.g., USD/INR)** | Exposure + Regime | Macro stress/Yield dynamics. |

**Portfolio Integration:**
The final portfolio return is a weighted average of the decisions made by the three individual asset models.

---

## 4. Technical Specification
### 4.1 Feature Engineering
All features are calculated at **Market Close ($T$)** and are stationary (scale-free).
* **Exposure Features:** `Ret_3d`, `Ret_5d`, `Ret_20d`, `Vol_Ratio`, `MA_Ratio`, `Close_Pos_Range`, `Intraday_Return`.
* **Regime Features:** `Vol_10d`, `Vol_20d`, `Abs_Return`, `Rolling_Range`.

### 4.2 Causal Labeling & Timing
To ensure no data leakage, the timing is strictly controlled:
1. **Decision Time:** $T$ Close (using features known at that moment).
2. **Entry:** $T+1$ Open (captures move *after* the overnight gap).
3. **Exit:** $T+4$ Close (total 3-day holding period).

**The Labeling Logic:**
$$\text{Label} = \begin{cases} 1 & \text{if } \frac{Close_{T+4} - Open_{T+1}}{Open_{T+1}} > 0 \\ 0 & \text{otherwise} \end{cases}$$

---

## 5. Experimental Framework
### Experiment 1: Model vs. Random (Monte Carlo)
We compare the trained XGBoost models against a **500-iteration Monte Carlo simulation**.
* **Model Path:** Uses XGBoost probability to decide Long or Flat.
* **Random Path:** Uses a `random` library (or manual coin-flip) for a 50/50 Long/Flat decision.
* **Metric:** Terminal wealth distribution. The model is "successful" if it outperforms 95% of the random runs.

### Experiment 2: Regime-Aware Weighting
We test if applying the **Regime Model** (Calm vs. Volatile) to adjust position sizes outperforms a static-weight portfolio.

---

## 6. Model Parameters (XGBoost)
To prevent overfitting to financial noise, the models are kept intentionally "shallow":
* **Max Depth:** 3
* **Learning Rate:** 0.05
* **Estimators:** 400
* **Validation:** `TimeSeriesSplit` (No random shuffling of time-series data).

---

## 7. Limitations & Scope
* **Execution:** Transaction costs and slippage are excluded to isolate the "mathematical edge."
* **Gap Risk:** The model does not predict the $T$ to $T+1$ gap, only the subsequent 3-day trend.
* **Scope:** This is a structural test of signal validity, not a production-ready trading system.

## 8. Conclusion
METR provides a disciplined environment to separate luck from skill. By fixing the holding period and using non-correlated assets, the study ensures that any recorded "outperformance" is a result of the model capturing persistent market microstructure rather than a single lucky trend.