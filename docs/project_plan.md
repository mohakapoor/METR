# METR — Multi-Phase Execution Plan

## Aim & Ideology

The fundamental research question driving METR is: 
**Can a machine learning model trained purely on historical price, volume, and publicly available implied volatility data — without any live sentiment, news, macro indicators, or order flow — actually beat random market entries in the long term?**

This project operates under strict data constraints: the models are completely blind to real-world context. They must attempt to extract a statistically significant predictive edge entirely from mathematical market microstructure (momentum, volatility regimes both realized and implied, and mean reversion patterns) and validate against 1,000 Monte Carlo simulations of random trade selection.

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
- [x] Refactor `src/data_cleaning.ipynb` to apply Triple Barrier logic programmatically.
- [x] Align time-series data perfectly across all 3 assets to prevent date mismatches.
- [x] Engineer relative-strength cross-asset features (e.g., Equity vs. Gold momentum, Cross-Vol Ratios).
- [x] **Integrate India VIX (`^INDIAVIX`):** Introduce forward-looking implied volatility to replace/supplement backward-looking realized volatility (`Vol_20d`).
- [x] Map labels to a 3-class target system (e.g., 0=Timeout, 1=Long, 2=Short) and export updated datasets.

### Phase 4: Multi-Asset Meta-Labeling ✅ *(Completed)*
- [x] Pivot from 3-class price prediction to Binary Meta-Labeling.
- [x] Implement the `Signal` + `Meta_Label` generation in `src/indicators.py`.
- [x] Identify $T=5$ as the optimal prediction horizon for return separation.

### Phase 5: Feature Separation & Baseline ✅ *(Completed)*
- [x] Develop `src/feature_separation.py` to identify "Green Light" indicators.
- [x] Establish a Linear Baseline using `src/train_baseline.py` (Logistic Regression).
- [x] Document the "Zero-Recall" wall on Nifty for linear models.

### Phase 6: Interaction Features & Macro Regime ✅ *(Completed)*
- [x] Engineer High-Conviction Interaction Features (`ATR_MACD`, `Vol Efficiency`, `RSI_Trend`).
- [x] Integrate **India VIX** for implied volatility regime detection.
- [x] Re-establish the **Interaction Baseline** (Logistic Regression + 5.0% ROC AUC boost).
- [x] Identify the "Non-Linear Requirement" for Nifty.

### Phase 7: Meta-Training & Filter Overhaul ✅ *(Completed)*
- [x] Overhaul `src/train_trade_filter.py` for **Binary Meta-Labeling**.
- [x] Implement **RandomizedSearchCV** (n_iter=20) for better coverage.
- [x] Integrate "Exhaustion Hypothesis" insights (depth/regularization).
- [x] Achieve statistically significant Alpha boost in **Gold** (WR 50%+) and **Nifty** (AUC 0.56+).

### Phase 8: Model Interpretation & Macro-Stabilization ✅ *(Completed)*
- [x] **SHAP/Gain Analysis**: Isolated `Vol Efficiency` and `VIX_Shock` as the primary Alpha drivers for Gold.
- [x] **Macro-Injection (Phase 8.6)**: Successfully bridged the "Interaction Wall" in USDINR by injecting equity-stress filters (+10.7% edge).
- [x] **Stabilization (Phase 8.7)**: Reached an **Institutional-Grade Alpha Peak** (+10.74% edge) while reducing the overfitting gap to 0.19.
- [x] **Final Alpha Audit (Phase 8.8)**: Mathematically confirmed the separation power of the production features.

### Phase 9: Regime Conditioning (GaussianHMM) ✅ *(Rejected)*
- [x] **Experimentation**: Fitted 3-state HMM to identify "Crisis" regimes.
- [x] **Verdict**: HMM rejected for inclusion; its "Crisis" detection was redundant and less precise than the existing **VIX_Shock** and **Nifty_Vol** indicators.

### Phase 10: Backtest & Economics 🚀 *(In Progress)*
- [x] **Cost Specification**: Integrated **10 bps (0.1%)** round-trip transaction costs as the production floor.
- [/] **Equity Curve Engine**: Developing `src/backtest_engine.py` to calculate P&L from the `TB_Return` column.
- [ ] **Risk Analytics**: Compute Sharpe, Max Drawdown, and Avg Hold Period for the "Golden Build."
- [ ] **Optimal Thresholding**: Identify the exact Recall/Sharpe trade-off for deployment.

### Phase 11: Statistical Validation (The Final Test)
- [ ] **Monte Carlo Baseline A**: Pure random trade selection (1,000 runs).
- [ ] **Monte Carlo Baseline B**: Signal-only (no model filter) baseline.
- [ ] **P-Value Analysis**: Calculate model Sharpe percentile ranking vs. MC distribution.
- [ ] **Walk-Forward 2025**: Final validation on OOS 2025-Q1 data.