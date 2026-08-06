<div align="center">
  <h1>METR</h1>
  <p><strong>Market Exposure Timing Research</strong></p>

  <h4>A personal research project exploring if machine learning can improve simple trading signals.</h4>

  <br />

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-00599C?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/XGBoost-1.7+-EE4C2C?style=for-the-badge&logo=xgboost&logoColor=white" />
    <img src="https://img.shields.io/badge/Polars-Data%20Engineering-F7D010?style=for-the-badge&logo=polars&logoColor=black" />
    <img src="https://img.shields.io/badge/Scikit--Learn-Analysis-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
    <img src="https://img.shields.io/badge/YFinance-Data-green?style=for-the-badge" />
  </p>

  <p align="center">
    <b>Status:</b> <code>Finished</code> &nbsp;•&nbsp; 
    <b>License:</b> <code>MIT</code>
  </p>
</div>

---

## What is this project?
> **"Can a machine learning model beat random market entries using just price and volume data?"**

METR is a personal experiment I built to see if I could use machine learning  to filter out bad trades from a basic momentum strategy. I wanted to test this rigorously, so I compared my model's performance against 10,000 random Monte Carlo simulations to make sure any success wasn't just luck.

---

## How it works

- **The Core Idea:** I start with a basic momentum signal (e.g., buy if the price has been going up). Then, I use an XGBoost classifier to look at other indicators and decide whether to actually take the trade or pass on it.
- **Labeling:** I used a "Triple Barrier" method to label my data. It sets a profit target, a stop loss, and a time limit (like 5 days) for each trade.
- **Features:** I used Polars to build technical indicators, including some fractional differentiation to make the data stationary while keeping its history.
- **Validation:** I checked the results using standard metrics like Sharpe ratio and ran Monte Carlo simulations to verify the stats.

---

## Results: GoldBees

The most interesting result came from the Gold ETF (GoldBees). The model was actually able to turn a losing basic momentum signal into a profitable one on unseen data from 2024–2025.

<div align="center">
  <img src="reports/backtest/gold_report.png" width="90%" alt="Gold Performance Report" />
</div>

### Performance Metrics (Out-of-Sample 2024–2025)
| Metric | Result |
| :--- | :--- |
| **Return** | +17.92% |
| **Sharpe Ratio** | 1.48 |
| **Max Drawdown** | 7.99% |
| **Win Rate** | 66.1% |

I ran a few statistical tests (like a Monte Carlo simulation with 10k runs), and the results for Gold were statistically significant (p-value ~0.01). However, the model didn't find any real edge for Nifty 50 or USD/INR, which makes sense given how efficient and managed those markets are.

---

## Project Structure

<details>
<summary>Click to view</summary>

```text
METR/
├── data/                    # Raw and processed data
├── docs/                    # My notes and methodology
├── src/
│   ├── fetch_data.py        # Fetches data from yfinance
│   ├── feature_eng.py       # Indicator logic using Polars
│   ├── triple_barrier.py    # Trade labeling logic
│   ├── train_trade_filter.py # The XGBoost model
│   ├── backtest_engine.py   # Code to calculate returns
│   └── monte_carlo_audit.py # Random simulations for testing
├── config.yaml              # Project settings
└── README.md                
```
</details>

---

## Documentation

If you want to read more about what I learned and how I built it:

1. **[Documentation & Methodology](docs/documentation.md):** The main write-up of my findings.
2. **[Observations](docs/observations.md):** My daily logs while working on the project.


### References
* **Triple Barrier Labelling Algorithm** by William Santos – *Implementation guidance for volatility-adaptive labeling.*
* **Fractional Differencing Financial Data** by The Quant Trading Room – *Theoretical foundation for memory preservation.*

---

<div align="center">
  <sub>Built by Mohak Kapoor</sub>
</div>
