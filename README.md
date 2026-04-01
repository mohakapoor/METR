# METR — Market Exposure Timing vs Randomness

## Overview & Ideology
**Can a machine learning model trained purely on historical price/volume data—without any live sentiment, news, macro indicators, or order flow—actually beat random market entries in the long term?**

**METR** is a controlled empirical study designed to isolate true statistical predictive edge from market noise and lucky streaks. By restricting the models to exclusively mathematical market microstructure (momentum, volatility regimes, mean reversion) and forcing a fixed 3-day holding period, the project evaluates whether standard assets (Equities, Commodities, FX) possess predictable short-term inefficiencies.

> **Current Status:** Phase 7 — **Meta-Training & Filter Overhaul**. We have synthesized **Interaction Features** (ATR_MACD, RSI_Trend) and integrated **India VIX** as a forward-looking regime filter. The current focus is on overhauling the XGBoost Meta-Model for Binary Meta-Labeling.

## Core Experiments
1. **Meta-Filter vs. Baseline (The Edge Test):** Evaluating our XGBoost "Trade Filter" against the Interaction-Enhanced Logistic Regression baseline. We measure "Meta-Precision" (Win Rate) to see if the model can successfully identify which momentum signals are trustworthy.
2. **Regime Detection (The Context Test):** Quantifying the impact of **India VIX** and **Cross-Asset Relative Strength** on signal quality—specifically focusing on the "Non-Linear Wall" in Nifty.

## Project Structure
```text
METR/
├── data/
│   ├── raw/                 # Raw OHLC parquet files (yfinance)
│   └── processed/           # Engineered features and Meta-Target labels
├── docs/                    # Extensive research notes, observations, and plans
│   ├── documentation.md     # In-depth findings and methodology 
│   ├── project_plan.md      # Chronological execution roadmap
│   └── observations.md      # Empirical results and Meta-Labeling logs
├── reports/
│   └── baseline/            # Logistic Regression benchmark results
├── models/                  # Exported .joblib model files
├── src/
│   ├── fetch_data.py        # Automated historical data ingestion pipeline
│   ├── data_cleaning.ipynb  # Join, align, and stabilize asset timestamps
│   ├── feature_engineering.ipynb # Advanced indicators and Meta-Label synthesis
│   ├── feature_eng.py       # Technical indicator library (Polars-optimized)
│   ├── tripple_barrier.py   # Forward-scanning dynamic labeling algorithm
│   ├── train_trade_filter.py # Primary XGBoost Meta-Model (GPU accelerated)
│   ├── train_baseline.py    # Logistic Regression baseline (StandardScaler)
│   ├── feature_separation.py # ROC AUC analysis of individual feature signal
│   └── feature_analysis.py  # Feature importance and Correlation extraction
├── config.yaml              # Global project config (features, thresholds, T=5)
└── README.md                # Project guide (this file)
```

## Features & Methodology
1. **Meta-Labeling Architecture:** Instead of direct direction prediction, we use a two-layer approach:
   - **Layer 1 (Signal)**: A 5-day momentum "Primary Boss" signal.
   - **Layer 2 (Filter)**: An XGBoost classifier that predicts `Meta_Label` (1 if Signal = Win, 0 otherwise).
2. **Adaptive Labeling (Triple Barrier):** We use a volatility-adaptive Triple Barrier sequence ($T=5$) with asset-specific $k$ values (Nifty: 1.5, Gold: 1.75, USDINR: 1.5).
3. **Fractional Differentiation ($d$):** Resolves the stationarity-memory trade-off by preserving memory (up to 91% correlation) while ensure statistical stationarity for the models.
4. **Strict Validation:** TimeSeriesSplit is strictly enforced across a 12-year window (2013-2025) to guarantee zero look-ahead bias.

## Getting Started

### Prerequisites
* Python 3.11+
* `polars`, `yfinance`, `xgboost`, `scikit-learn`, `matplotlib`, `pyyaml`, `statsmodels`

### Diving Deeper
Research observations and results are logged chronologically:

1. **[Documentation & Methodology](docs/documentation.md):** The core findings and theoretical foundations.
2. **[Grid Search Observations](docs/observations.md):** Daily logs and empirical results for Meta-Labeling.
3. **[Execution Roadmap](docs/project_plan.md):** Current progress and upcoming phases.

## References
* [Triple Barrier Labelling Algorithm](https://williamsantos.me/posts/2022/triple-barrier-labelling-algorithm/) by William Santos – *Implementation guidance for the forward-scanning volatility-adaptive labeling method.*
