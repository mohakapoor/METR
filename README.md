# METR — Market Exposure Timing Research

## The Question
**Can a machine learning model trained purely on historical price, 
volume, and publicly available implied volatility data — without 
any live sentiment, news, macro indicators, or order flow — 
actually beat random market entries in the long term?**

METR is a controlled empirical study testing whether three 
uncorrelated Indian asset classes possess predictable short-term 
inefficiencies exploitable without privileged data access. Edge 
is validated not against a passive benchmark, but against 10,000 
Monte Carlo simulations of random trade selection — isolating 
true statistical skill from market drift and lucky streaks.

---

## Methodology
The framework operates in two layers:

**Layer 1 — Base Signal:** A momentum conviction filter marks 
trading opportunities when 5-day return exceeds 20-day volatility. 
Days without sufficient conviction are marked flat (no trade).

**Layer 2 — Meta-Filter:** An XGBoost classifier trained on 
~30 price, volatility, and cross-asset features decides whether 
to execute each signaled trade. The model never predicts market 
direction — it only filters which signals are worth acting on.

Labels are generated via the Triple Barrier Method: if the signal 
direction matches the barrier outcome (take-profit or stop-loss 
hit), the trade is marked as a success. The model learns to 
identify conditions where momentum signals follow through.

**Assets:** Nifty 50, GoldBees (NSE), USD/INR  
**Training:** 2014–2023 (2012–2013 reserved as lookback buffer) | **Out-of-Sample:** 2024–2025 
**Features:** Momentum, volatility regimes, mean reversion, 
India VIX derivatives, cross-asset stress filters  

---

## Results

### Gold (GoldBees) — Statistically Significant Edge
| Metric | Value |
|--------|-------|
| Out-of-Sample Return | +17.92% |
| Sharpe Ratio | 1.48 (Account) / 1.51 (Signal) |
| Max Drawdown | 7.99% |
| Calmar Ratio | 1.87 |
| Win Rate | 66.1% (39/59 trades) |
| Raw Signal Baseline Sharpe | -0.73 |

The raw momentum signal alone loses money. The meta-filter 
transforms this into a profitable strategy — a +2.21 Sharpe 
lift attributable entirely to model trade selection.

### Statistical Validation (Gold, OOS 2024–2025)
| Test | P-value | Result |
|------|---------|--------|
| Monte Carlo (10,000 simulations) | 0.0104 | ✅ Significant |
| Binomial (win rate vs 50%) | 0.0092 | ✅ Significant |
| Kupiec (reliability) | 0.0126 | ✅ Significant |
| T-test (mean return) | 0.1099 | ❌ Underpowered* |

*T-test lacks power at n=59 trades due to selective filtering. 
Three independent tests confirm significance.

### Nifty 50 — Null Result
Model Sharpe -0.40. All four tests non-significant. Consistent 
with deep institutional coverage limiting price-only 
predictive power.

### USD/INR — Null Result  
Model Sharpe -0.60. All four tests non-significant. RBI 
intervention disrupts momentum patterns — structural ceiling 
on technical prediction for managed currencies.

---

## Core Finding
The asymmetric results across three asset classes are not a 
failure — they are the finding. The framework correctly identifies 
exploitable structure where theory predicts it (commodity ETF 
with embedded currency exposure) and correctly finds nothing 
where theory predicts absence of edge (benchmark equity index, 
managed currency). A framework producing uniformly positive 
results across all assets would be more suspicious, not more 
convincing.

> **Price, volume, and publicly available implied volatility 
> data are sufficient to construct a statistically significant 
> meta-filter on GoldBees that beats random market entry. 
> The same data is insufficient on India's benchmark equity 
> index or its managed currency pair.**

---

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
│   ├── triple_barrier.py   # Forward-scanning dynamic labeling algorithm
│   ├── train_trade_filter.py # Primary XGBoost Meta-Model (GPU accelerated)
│   ├── train_baseline.py    # Logistic Regression baseline (StandardScaler)
│   ├── backtest_engine.py   # Compounded equity curve and performance metrics
│   ├── monte_carlo_audit.py # 10,000-iteration statistical validation
│   ├── threshold_optimizer.py # Calmar-based threshold selection (train set only)
│   ├── shap_audit.py        # SHAP feature importance and attribution
│   ├── feature_separation.py # ROC AUC analysis of individual feature signal
│   └── feature_analysis.py  # Feature importance and Correlation extraction
├── config.yaml              # Global project config (features, thresholds, T=5)
└── README.md                # Project guide (this file)
```

### Diving Deeper
Research observations and results are logged chronologically:

1. **[Documentation & Methodology](docs/documentation.md):** The core findings and theoretical foundations.
2. **[Observations](docs/observations.md):** Daily logs and empirical results for the Project.

## References
* [Triple Barrier Labelling Algorithm](https://williamsantos.me/posts/2022/triple-barrier-labelling-algorithm/) by William Santos – *Implementation guidance for the forward-scanning volatility-adaptive labeling method.*
* [Is Differencing Too Much? Fractional Differencing Financial Data](https://medium.com/@The-Quant-Trading-Room/is-differencing-too-much-fractional-differencing-financial-data-03299c824c0d) by The Quant Trading Room – *Theoretical foundation for the Fractional Differentiation approach used in METR for memory preservation.*
