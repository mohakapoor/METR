# METR — Strategic Insights for Meta-Labeling

This document synthesizes our empirical findings from Phases 1–6 and outlines the strategic roadmap for the **Phase 7: Meta-Training (XGBoost)** overhaul. Use these insights to guide hyperparameter selection and model calibration.

---

## 1. The "Exhaustion" Hypothesis (Critical)
Our latest `feature_separation.py` runs revealed a counter-intuitive but powerful pattern in **Nifty** and **Gold**:
*   **Observation**: Higher `Vol Efficiency` (Clean, low-volatility price moves) actually correlates with **Signal Failure**.
*   **Insight**: In Equities and Commodities, vertical price moves without intraday "breathers" are often signs of climax/exhaustion. 
*   **XGBoost Strategy**: The model must use `max_depth >= 3` to resolve the non-linear "bend" where a trend is good until it becomes *too* efficient, at which point it becomes a trap.

## 2. Cross-Asset Regime Anchors
The following features have the highest separation power and should be the "anchors" for our meta-filter:
*   **`Gold_Nifty_RS_5d`**: Risk-On/Risk-Off detector. Nifty momentum is only reliable when it's outperforming Gold. 
*   **`VIX_Relative`**: Measures the "Fear Premium." If VIX is extremely high relative to its 20-day average, the market is usually in a panic-bottom or a high-stress reversal zone.
*   **`Usdinr_Stress_Filter`**: Currency volatility is a "hidden" pressure on Indian Equities. If USDINR is spiking (`Ret_5d`), Nifty momentum signals should be treated with extreme skepticism.

## 3. Asset-Specific Personalities
We have learned that **USDINR** behaves differently from **Nifty/Gold**:
*   **USDINR**: "Efficient" moves tend to **persist**. Clean trends are winners.
*   **Nifty/Gold**: "Efficient" moves tend to **exhaust**. Clean trends are traps.
*   **Strategic Action**: Consider using different `reg_lambda` (L2 regularization) for USDINR vs. Equities to account for the difference in trend persistence.

## 4. XGBoost Hyperparameter Recommendations
Based on the current 3412-row dataset and ~40 interaction features:

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| `learning_rate` | **0.01 – 0.03** | We have a small dataset; slow learning prevents over-fitting to noisy interaction outliers. |
| `max_depth` | **3 – 5** | Needed to capture the interaction between `VIX` and `Vol Efficiency`. |
| `colsample_bytree` | **0.7 – 0.8** | Forces trees to "discover" signals beyond just the top-ranked features. |
| `subsample` | **0.8** | Standard protection against row-level noise. |
| `scale_pos_weight` | **1.0** | Our `Meta_Label` distribution is fairly balanced (~42% wins); no need for heavy weighting. |

## 5. The "Win-Rate First" Mandate
In Meta-Labeling, **Precision is the only metric that matters.**
*   **Target**: Win Rate > 55% (Nifty/USDINR) and > 65% (Gold).
*   **Trade Fraction**: It is acceptable to have a low trade fraction (5–15%) if the precision is high. 
*   **Calibration**: Use the `THRESHOLD` sweep (0.52 to 0.58) to aggressively kill low-conviction trades. We are building a "Trade Filter," not a "Trade Generator."

---

---

## 7. Phase 8 Post-Audit Blueprints (Portfolio Alpha)

The SHAP/Gain audit has revealed the specific non-linear "Blueprints" that allow our meta-filters to outperform random chance.

### A. The Gold Blueprint: "The Microstructure Chaos Filter"
*   **Discovery**: **`Vol Efficiency`** is the #1 signal (SHAP 0.26).
*   **The Logic**: Gold momentum is only reliable when it is "messy" or "inefficient." When the price moves in a clean, vertical line, the model identifies it as an **Exhaustion Point** and filters the signal.

### B. The Nifty Blueprint: "The Memory-Extension Filter"
*   **Discovery**: **`FD_Close`** (Long-Memory) is the master anchor.
*   **The Logic**: Nifty momentum fails when the absolute extension from the mean (**`Price_vs_MA20`**) is too high. The model uses the fractionally differentiated price to find stationary points where momentum is likely to mean-revert rather than trend.

### C. The USDINR Blueprint: "The VIX Proxy"
*   **Discovery**: The model is 38% correlated with **VIX Intensity**.
*   **The Logic**: USDINR momentum is almost entirely a "Beta" play on Global Stress. If VIX is spiking, local currency technicals are secondary to global macro-flow.

---

## Final Milestone: Phase 9 Starting Position
With these Blueprints identified, the next step is **GaussianHMM Regime Conditioning**. Instead of letting the trees "guess" if the trend is efficient or extended, the HMM will explicitly tag the current state (Bull Trend, Bear Trend, or Choppy), allowing the meta-filters to calibrate their confidence dynamically.
