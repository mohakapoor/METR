# METR — Market Exposure Timing vs Randomness

## Overview & Ideology
**Can a machine learning model trained purely on historical price/volume data—without any live sentiment, news, macro indicators, or order flow—actually beat random market entries in the long term?**

**METR** is a controlled empirical study designed to isolate true statistical predictive edge from market noise and lucky streaks. By restricting the models to exclusively mathematical market microstructure (momentum, volatility regimes, mean reversion) and forcing a fixed 3-day holding period, the project evaluates whether standard assets (Equities, Commodities, FX) possess predictable short-term inefficiencies.

> **Current Status:** Phase 3 — **Cross-Asset Feature Engineering & Data Pipeline**. Optimization of **Fractional Differentiation** and **India VIX** parameters is complete. Integration of cross-market relative strength features is underway before retraining a Unified Multi-Class machine learning model.

## Core Experiments
1. **Model vs. Random (The Edge Test):** Evaluating the model's predictive capabilities against a massive 1,000-iteration Monte Carlo simulation of purely random (coin-flip) trading strategies to ensure genuine statistical outperformance.
2. **Unified vs. Isolated Modeling (The Context Test):** Comparing single-asset models (e.g., Nifty predicting Nifty) against a Unified cross-asset model to see if broader macro-context can be inferred solely from relative price action.

## Project Structure
```text
METR/
├── data/
│   ├── raw/                 # Raw OHLC parquet files (yfinance)
│   └── processed/           # Engineered features and Target labels
├── docs/                    # Extensive research notes, observations, and plans
│   ├── documentation.md     # In-depth findings and methodology 
│   ├── project_plan.md      # Chronological execution roadmap
│   └── observations.md      # Empirical grid search results and tuning logs
├── src/
│   ├── fetch_data.py        # Automated historical data ingestion pipeline
│   ├── data_cleaning.ipynb  # Join, align, and stabilize asset timestamps
│   ├── feature_engineering.ipynb # Advanced indicators, labels, and training set synthesis
│   ├── tripple_barrier.py   # Forward-scanning dynamic labeling algorithm
│   ├── train_exposure.py    # XGBoost training & regularization 
│   ├── evaluate_model.py    # Generates diagnostic plots (ROC, Confusion Matrix)
│   ├── benchmark_random.py  # Monte Carlo simulation engine
│   └── feature_analysis.py  # Importance and Correlation extraction
├── config.yaml              # Global project config (features, thresholds, weights)
└── README.md                # Project guide (this file)
```

## Features & Methodology
1. **Adaptive Labeling (Triple Barrier Method):** Instead of forcing naive Up/Down binary predictions, METR uses a volatility-adaptive Triple Barrier sequence ($T=3$, $k=1.5\sigma$) to filter out non-directional chop (Label `0`). Computed in `src/feature_engineering.ipynb`.
2. **Fractional Differentiation ($d$):** Resolves the stationarity-memory trade-off by preserving up to 91% of historical memory while reaching statistical stationarity. Implemented in `src/feature_engineering.ipynb`.
3. **Strict Validation:** TimeSeriesSplit is strictly enforced across a 12-year window (2013-2025) to guarantee zero look-ahead bias.
4. **Implied Volatility (India VIX):** Captures forward-looking market expectations and fear-gauges as a primary predictive feature.

## Getting Started

### Prerequisites
* Python 3.10+
* `polars`, `yfinance`, `xgboost`, `scikit-learn`, `matplotlib`, `pyyaml`, `statsmodels`

### Diving Deeper
This repository is primarily structured as a chronological  study.You can explore the exact methodology, grid search results, and logic shifts by reading the detailed notes:

1. **[Documentation & Methodology](docs/documentation.md):** The core findings, architectures, and theoretical foundations of the METR experiment.
2. **[Grid Search Observations](docs/observations.md):** Detailed logs, logic patches, and empirical parameters chosen specifically for each asset.
3. **[Execution Roadmap](docs/project_plan.md):** The chronological phases tracking the project's progression.

## References
* [Triple Barrier Labelling Algorithm](https://williamsantos.me/posts/2022/triple-barrier-labelling-algorithm/) by William Santos – *Implementation guidance for the forward-scanning volatility-adaptive labeling method.*
