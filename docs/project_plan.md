# METR — Multi-Phase Execution Plan

## Aim & Ideology

The fundamental research question driving METR is: 
**Can a machine learning model trained purely on historical price/volume data—without any live sentiment, no news, no macro indicators, and no order flow—actually beat random market entries in the long term?**

This project operates under strict data constraints: the models are completely blind to real-world context. They must attempt to extract a statistically significant predictive edge entirely from mathematical market microstructure (momentum, volatility regimes—both realized and implied—and mean reversion patterns).

## Core Experiments
To empirically test this ideology, the project is structured around two core experiments running simultaneously across three uncorrelated asset classes (Equities, Commodities, FX):

1. **Model vs. Random (The Edge Test):** Comparing the models' predictive trades over a fixed 3-day holding period against a massive Monte Carlo simulation of 1,000 purely random (coin-flip) trading strategies. This isolates true statistical skill from lucky streaks or general market drift.
2. **Unified vs. Isolated Modeling (The Context Test):** Comparing models trained exclusively on single-asset features (e.g., Nifty predicting Nifty) against Unified cross-asset models (e.g., Gold volatility mapped against Nifty momentum) to determine if macro-context can be inferred purely from relative price action.

---
### Phase 1: Foundation & Initial Modeling ✅ *(Completed)*
- [x] Fetch OHLC data for Equity, Commodity, and FX markets.
- [x] Engineer foundational technical features (Momentum, Trend, Volatility).
- [x] Train baseline binary predictive models (Model v1).
- [x] Implement aggressive regularization to counter overfitting (Model v2).
- [x] Build and run the 1000-iteration Monte Carlo random baseline.

### Phase 2: Label Optimization (Triple Barrier) ✅ *(Completed)*
- [x] Implement forward-scanning dynamic Triple Barrier logic.
- [x] Run comprehensive grid-search validation across Nifty, Gold, and USDINR.
- [x] Calibrate intraday tie-breaker edge cases to prevent false stop-outs.
- [x] Shift evaluation metric from pure balance to tradeable Return Spread.
- [x] Finalize optimal $k$ scaling for $T=3$ holding periods across all assets.

### Phase 3: Data Pipeline & Advanced Feature Engineering ✅ *(Completed)*
*Improved labels alone without improved features will not yield significantly better predictive power. Cross-market intelligence must be injected before retraining.*
- [x] Refactor `src/data_cleaning.ipynb` to apply Triple Barrier logic programmatically.
- [x] Align time-series data perfectly across all 3 assets to prevent date mismatches.
- [x] Engineer relative-strength cross-asset features (e.g., Equity vs. Gold momentum, Cross-Vol Ratios).
- [x] **Integrate India VIX (`^INDIAVIX`):** Introduce forward-looking implied volatility to replace/supplement backward-looking realized volatility (`Vol_20d`).
- [x] Map labels to a 3-class target system (e.g., 0=Timeout, 1=Long, 2=Short) and export updated datasets.

### Phase 4: Multi-Asset Meta-Labeling ✅ *(Completed)*
- [x] Pivot from 3-class price prediction to Binary Meta-Labeling.
- [x] Implement the `Signal` + `Meta_Label` generation in `src/feature_eng.py`.
- [x] Identify $T=5$ as the optimal prediction horizon for return separation.

### Phase 5: Feature Separation & Baseline ✅ *(Completed)*
- [x] Develop `src/feature_separation.py` to identify "Green Light" indicators.
- [x] Establish a Linear Baseline using `src/train_baseline.py` (Logistic Regression).
- [x] Document the "Zero-Recall" wall on Nifty for linear models.

### Phase 6: Meta-Training & Filter Optimization
- [ ] Train the primary XGBoost Meta-Model using `train_trade_filter.py`.
- [ ] Optimize the model for **Meta-Precision** (Win Rate > 55%).
- [ ] Calibrate the `THRESHOLD` lever for Nifty, Gold, and USDINR.
- [ ] Final evaluation vs. the Logistic Regression baseline.