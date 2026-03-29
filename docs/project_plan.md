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

### Phase 3: Data Pipeline & Advanced Feature Engineering ⏳ *(Up Next)*
*Improved labels alone without improved features will not yield significantly better predictive power. Cross-market intelligence must be injected before retraining.*
- [ ] Refactor `src/data_cleaning.ipynb` to apply Triple Barrier logic programmatically.
- [ ] Align time-series data perfectly across all 3 assets to prevent date mismatches.
- [ ] Engineer relative-strength cross-asset features (e.g., Equity vs. Gold momentum, Cross-Vol Ratios).
- [ ] **Integrate India VIX (`^INDIAVIX`):** Introduce forward-looking implied volatility to replace/supplement backward-looking realized volatility (`Vol_20d`).
- [ ] Map labels to a 3-class target system (e.g., 0=Timeout, 1=Long, 2=Short) and export updated datasets.

### Phase 4: Model Retraining (Multi-Class) 
- [ ] Update `config.yaml` to handle multi-class XGBoost parameters and new features.
- [ ] Train the new 3-class models on the Triple Barrier labels.
- [ ] Evaluate model output probabilities against the new $T=3$ Return Spread.
- [ ] Compare isolated models (Nifty-only features predicting Nifty) vs Unified models (all features predicting Nifty).

### Phase 5: Regime & Unified Modeling
- [ ] Implement Regime Modeling (Calm vs Volatile) to dynamically size model convictions.
- [ ] Test regime-aware weighting across the entire unified portfolio.

### Phase 6: Final Evaluation & Documentation
- [ ] Re-run Monte Carlo random baseline against the updated 3-class model.
- [ ] Synthesize empirical findings on predictive ceilings across different asset classes.
- [ ] Finalize `documentation.md`.