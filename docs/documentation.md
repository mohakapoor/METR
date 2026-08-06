# METR — Documentation & Findings

> **Market Exposure Timing vs Randomness**
> A personal project to see if machine learning models can beat random market entries using basic price data.

---

## 1. Project Overview

### Objective
I wanted to see if machine learning models trained on price, volume, and implied volatility (VIX) could figure out when a basic momentum signal is actually worth trading. I used a technique called meta-labeling and tested the results against 10,000 random Monte Carlo simulations to make sure any success wasn't just luck.

### How It Works (The Two-Layer Setup)
- **Layer 1 (The Trigger)**: A basic 5-day momentum signal generated in `src/indicators.py`.
- **Layer 2 (The Filter)**: An XGBoost classifier (`src/train_trade_filter.py`) that acts as a second opinion, deciding whether to take the trade based on broader market conditions.
- **Testing**: I used `src/monte_carlo_audit.py` to check for statistical significance.
- **Assets tested:** Nifty 50, GOLDBEES (NSE), and USD/INR.
- **Validation method:** 5-fold TimeSeriesSplit, and I only evaluated the model on days where a trade was actually triggered ($Signal \neq 0$).

### Tech Stack
| Component | What I Used | Why |
|-----------|--------|----------|
| **Data** | Polars | Fast execution and joins |
| **Model** | XGBoost | Standard binary classification |
| **Labeling** | Triple Barrier | Uses volatility to set dynamic profit/loss targets |
| **Memory** | FracDiff | Fractional differentiation ($d=0.45$) to keep historical memory |
| **Metrics** | Calmar / Sharpe | Adjusted for 5-bps trade friction |

---

## 2. Data Pipeline

### Source
- I fetched raw OHLCV data using `src/fetch_data.py`.
- The assets are Nifty, GOLDBEES, USD/INR, and **India VIX** (a volatility gauge).
- The total row counts varied slightly because different markets have different holidays.

### Data Pipeline Stages
I split the data work across two notebooks:

#### 2.1 Alignment (`src/data_cleaning.ipynb`):
- **Syncing Assets**: I aligned Nifty, Gold, USD/INR, and VIX to the same timeline.
- **Calendar Alignment**: I handled the mismatched market holidays (like MCX vs. NSE).
- **Date Intersection**: I made sure all cross-asset features were only calculated on days where all markets were open (around 3,440+ days).
- **Lookback Buffer**: I used 2012-2013 data just to warm up the indicators, and started the actual training data from 2014-01-01.

#### 2.2 Feature Engineering (`src/feature_engineering.ipynb`)
- **Fractional Differentiation**: I used this (`src/frac_diff.py`) to make the data stationary for the ML model while keeping about 80% of its historical memory.
- **Labeling Target**: I generated targets using the Triple Barrier method with a 5-day window ($T$) and different volatility multipliers ($k$) for each asset.
- **Directional Returns**: I adjusted the returns based on the trade signal (`Return * Signal`) so the model knew if a positive outcome was from a long or short trade.
- **Stress Features**: I created features like `Usdinr_Stress_Filter` to capture how different assets interacted during market stress.

### 2.3 Base Signal 
The primary momentum signal is constructed as:
- **Signal = +1** if $Ret_{5d} > Vol_{20d}$ (Long conviction)
- **Signal = -1** if $Ret_{5d} < -Vol_{20d}$ (Short conviction)
- **Signal = 0** otherwise (Flat — no trade)

**Important constraint**: The XGBoost filter is only trained and tested on rows where $Signal \neq 0$. It doesn't try to predict every day, just the days a trade is suggested.

**Total Features:** 13 Technical + 12 Interaction + 5 Macro/VIX.

### Train/Test Split
- **Buffer Data:** 2012–2013 (For rolling indicators)
- **Train (In-Sample):** 2014–2023 (~2200 rows)
- **Test (Out-of-Sample):** 2024–2025 (~450 rows)
- I split this strictly by time to avoid data leakage.

---

## 3. Model Development

I chose XGBoost because it usually works best for tabular financial data. Deep learning models like LSTMs need way more data than I had available and tend to overfit small datasets.

---

### Model v1 — Initial Attempt (Too Much Overfitting)

**Best Parameters Found:**
```
learning_rate: 0.1, max_depth: 4, n_estimators: 5000, subsample: 1.0
```

**Results:**

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 86.54% | 53.42% |
| ROC AUC | — | 0.5251 |
| **Overfit Gap** | **33.1%** | ⚠️ Way too high |

**Analysis:** The model was heavily overfitting. An 86% training accuracy meant it was memorizing the past data rather than learning real patterns. 

---

### Model v2 — Tuning Regularization

**Results after tuning:**

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 59.24% | 51.66% |
| ROC AUC | — | 0.5020 |
| **Overfit Gap** | **7.6%** | ⚠️ Much better |

**Diagnosis:** I fixed the overfitting (gap dropped from 33% to 7.6%), but my initial regularization was too strict (`reg_alpha=1.0`, `reg_lambda=5.0`), which killed the model's ability to find any signals at all. I had to dial it back to lighter values (`reg_alpha=0.1`, `reg_lambda=0.5`) to strike a balance.

---

### Model v3 — Handling Direction

**Improvements:**
1. **Sign-Adjusted Returns**: I changed the target to `Directional_Return = TB_Return * Signal` so the model understood if it was winning on a long or short trade.
2. **Filtering**: I strictly trained only when $Signal \neq 0$.
3. **Cross-Validation**: I started using 5-fold TimeSeries Cross-Validation to get a more realistic estimate of the probabilities.

**Results (Model v3):**

| Asset | ROC AUC | AUPRC |
|-------|---------|-------|
| **Nifty** | **0.5918** | 0.4431 |
| **Gold** | **0.5935** | 0.4602 |
| **USD/INR** | 0.5489 | 0.4833 |

**Conclusion:** Fixing the directional logic helped uncover some edge that was hidden before. 

---

## 4. Benchmark Results (Model vs Random)

### Monte Carlo Simulation
I ran 10,000 simulations where trades were picked at random dates (but matching the frequency of my model) to see if my model was actually doing anything special.

**Results (Early Model):**

| Metric | Model | Random Average |
|--------|-------|----------------|
| Total Return | 18.73% | 14.90% |
| Win Rate | 60.5% | ~51% |
| P-Value | 0.376 | — |

**Interpretation:**
- My model beat the random average by about 3.8%.
- A p-value of 0.38 means this result wasn't statistically significant yet.
- However, a 60.5% win rate compared to a 51% random baseline was a good sign.
*(Note: This was on training data. The final test results are in Section 15).*

---

## 5. Feature Importance Analysis

I found that different assets rely on very different features to predict success. Here's what worked for each:

### 5.1 Nifty 50 (Equities)
*Nifty relies mostly on relative strength against Gold and general volatility.*

| Feature | Importance (Gain) | Correlation (r) |
|---------|-------------------|-----------------|
| `Gold_Nifty_RS_5d` | 6.44 | +0.128 |
| `ROC_10` | 6.38 | -0.077 |
| `VIX_ATR_Ratio` | 5.21 | -0.128 |

### 5.2 Gold
*Gold was heavily driven by volatility and fear in the market.*

| Feature | Importance (Gain) | Correlation (r) |
|---------|-------------------|-----------------|
| `VIX_Momentum_Efficiency`| **13.61** | +0.007 |
| `RSI` | 11.06 | +0.031 |
| `RS_Momentum_Decoupling` | 2.76 | **-0.108** |

### 5.3 USD/INR (Currency)
*The currency model looked mostly for signs of stress in the broader Indian equity market.*

| Feature | Importance (Gain) | Correlation (r) |
|---------|-------------------|-----------------|
| `Vol Efficiency` | 2.83 | **+0.159** |
| `Nifty_Vol_Ratio` | 2.71 | **+0.153** |
| `Usdinr_Stress_Filter` | 4.11 | +0.048 |

---

## 6. What Didn't Work

| Approach | Why It Failed |
|----------|--------------|
| **USD/INR Zero-Fee Model** | It looked great on paper, but once I added a realistic 5-bps trading fee, the edge disappeared completely. |
| **Hidden Markov Models** | I tried using an HMM to detect "crisis" states, but it didn't work any better than just looking at simple VIX spikes. |
| **Long-Only Labels** | Initially, I ignored short trades. This confused the model during pullbacks. |
| **Basic RSI/MACD** | On a 5-day horizon, standard indicators didn't have enough predictive power on their own. |

---

## 7. What Did Work

| Finding | Impact |
|---------|----------|
| **Directional Labels** | Factoring in the trade direction (long/short) helped isolate the edge. |
| **Cross-Asset Features** | Comparing Nifty vs. Gold improved the correlations nicely. |
| **Testing with Friction** | Adding a 5-bps cost helped me realize Gold was the only asset worth trading. |
| **Fractional Differencing** | Keeping 80% of the historical memory ($d=0.45$) proved much better than standard differencing. |

---

## 8. Finalized Labeling: Triple Barrier Method
*(Code: `src/feature_engineering.ipynb`)*

Instead of just labeling days as "up" or "down", I used a rolling 5-day window ($T$) with volatility-based targets.
- **Upper Barrier:** Price hits profit target of $k$ * daily volatility (Label = +1)
- **Lower Barrier:** Price hits stop-loss of $k$ * daily volatility (Label = -1)
- **Time Barrier:** 5 days pass without hitting either (Label = 0)

### 8.1 Handling Trade Direction
If the base signal is a short trade, hitting the lower barrier is a win. I adjusted for this using:
```python
Directional_Return = TB_Return * Signal
Meta_Label = 1 if Directional_Return > 0 else 0
```

### 8.2 Using a 5-Day Window
I found that a 3-day window ($T=3$) caught too much random daily noise. Moving to a 5-day window ($T=5$) gave the trades enough time to play out and created a better spread between winning and losing trades.

### 8.3 Asset Multipliers ($k$)
- **NIFTY**: $k=1.5$
- **GOLD**: $k=1.75$ (Needs wider stops due to volatility)
- **USDINR**: $k=1.5$

---

## 9. Fractional Differentiation
*(Code: `src/feature_engineering.ipynb`)*

To make price data work with ML models, you usually have to difference it (e.g., use daily returns). The problem is this wipes out all the long-term price memory. I used fractional differencing to find a middle ground. 

I manually picked the differencing value ($d$) that kept as much correlation with the original price as possible while still being mostly stationary.

| Asset | Chosen $d$ | Correlation Kept | ADF p-value | Why I picked it |
|---|---|---|---|---|
| **Nifty 50** | **0.45** | 0.818 | 0.079 | Good balance of memory and stationarity |
| **Gold** | **0.50** | 0.772 | 0.096 | Kept it from decaying too much |
| **USD/INR** | **0.30** | 0.908 | 0.034 | Easily became stationary while keeping 91% memory |

---

## 10. Key Features Used

Here are some of the main custom features I built:

### Momentum & Volatility
- **`Vol Efficiency`**: `Ret_5d / Vol_20d`. Shows how strong a trend is compared to recent volatility.
- **`FD_Close`**: The fractionally differenced price.

### Cross-Asset Indicators
- **`Relative Strength (RS)`**: `Gold_Ret_5d - Nifty_Ret_5d`. Shows where money is flowing.
- **`Usdinr_Stress_Filter`**: `Risk_Off * Nifty_Vol_Ratio`. Looks for times when equity markets are stressed and money is fleeing to safety.
- **`Cross_Vol_Ratio`**: `Nifty_Vol_20d / Gold_Vol_20d`. 

### VIX (Fear Gauge)
- **`VIX_Relative`**: `VIX / SMA(VIX, 20)`. Checks if current fear is above the monthly average.
- **`VIX_ATR_Ratio`**: Checks if implied volatility (VIX) is getting disconnected from actual realized volatility (ATR).
- **`VIX_Momentum_Efficiency`**: Identifies price trends that are continuing *despite* rising fear in the market.

---

## 11. Finding the Optimal Threshold 
*(On 2014–2023 Training Data)*

I tested different probability thresholds to find the best balance of returns and drawdowns, assuming a 5 basis point trading fee.

| Asset | Chosen Threshold | Net Sharpe | Net Max Drawdown | Outcome |
|-------|------------------|------------|------------------|---------------|
| **GOLDBEES** | **0.52** | **1.42** | **18%** | **Looks good** |
| **Nifty 50** | **0.49** | **0.63** | **7%** | **Marginal** |
| **USD/INR** | 0.50 | **-2.41** | 3% | **Failed due to fees** |

For Nifty, setting the threshold to 0.49 instead of 0.48 dropped the maximum drawdown significantly (from 23% to 7%), which felt like a much safer bet.

---
## 12. Model Stability Updates

I realized relying just on technicals caused the model to vary too much. Adding macro features (like VIX and cross-asset comparisons) helped ground the model. 

For instance, the SHAP values showed that Gold relies heavily on `Vol_Efficiency` and `VIX_Momentum_Efficiency`. The model learned that Gold momentum works best when the rest of the market is stressed.

---

## 13. Out-of-Sample Results (2024–2025)

I ran the final test on unseen data from 2024 to 2025, using a 5 bps fee. 

| Asset | Target Threshold | Account Sharpe | Max Drawdown | Verdict |
|---|---|---|---|---|
| **GOLDBEES** | **0.52** | **1.48** | **8%** | **Statistically Significant** |
| **Nifty 50** | **0.49** | **-0.48** | **16%** | **No Edge Found** |
| **USD/INR** | **0.50** | **-0.80** | **5%** | **No Edge Found** |

Ultimately, Gold was the only asset worth keeping in the strategy. USD/INR and Nifty just didn't hold up after costs.

---

## 14. Failed Experiment: Hidden Markov Models

I tried building a Hidden Markov Model (HMM) to explicitly detect "crisis" market states. It ended up just being overly complicated and didn't perform any better than my simpler VIX indicators, so I scrapped it.

---

## 15. Statistical Validation

To make sure the Gold results weren't just a fluke, I ran three different statistical tests against the Monte Carlo random simulations.

| Test | Gold P-value | Nifty P-value | USD/INR P-value |
|---|---|---|---|
| **Monte Carlo (10k runs)** | **0.0104** | 0.4019 | 0.1237 |
| **Binomial (Win Rate)** | **0.0092** | 0.4721 | 0.3725 |
| **Kupiec** | **0.0126** | 0.8886 | 0.6841 |

*(Note: The Sharpe ratio used for the Monte Carlo comparison was slightly different than the account Sharpe, as it only annualized based on trade days).*

**Conclusion**: Gold passed all the statistical tests, confirming the model actually learned a real edge for that asset. 

---

## 16. Final Conclusions

### 16.1 Main Finding
My XGBoost model successfully improved a basic momentum strategy for the Gold ETF (GOLDBEES) on unseen data. It proved that it could filter out bad trades better than random chance (with p-values < 0.05). 

### 16.2 Gold Results
```
Total Return:     +17.92% (out-of-sample)
Account Sharpe:   1.48
Max Drawdown:      7.99%
Win Rate:          66.1% (39 out of 59 trades)
Baseline Sharpe:  -0.73 (If traded without the ML filter)
```
The raw momentum signal was losing money (Sharpe -0.73). The ML filter saved it, bumping the Sharpe up to +1.48. This proved my core idea worked: use simple momentum for the trigger, and machine learning for the filter.

### 16.3 Nifty 50 and USD/INR
The model couldn't find a reliable edge for Nifty 50 or USD/INR. 
- Nifty is highly efficient and heavily traded, so simple price data isn't enough to beat it.
- USD/INR is managed by the RBI, meaning its price moves are often dictated by central bank intervention rather than natural momentum.

I think it's a good sign that the model failed here—it shows it wasn't just overfitting to everything. It only found an edge where it structurally made sense (a commodity ETF).

---

## 17. Limitations

1. **Sample Size**: I only had 59 trades for Gold in the test period. It's a small sample size, which limits how confident I can be.
2. **Train/Test Split**: I picked my thresholds (0.52 for Gold) using the training data, and while it held up in the test data, having a third validation set would have been more rigorous.
3. **Stationarity vs Memory**: My fractional differencing values left the data slightly non-stationary (p-values around 0.07 - 0.09). I chose to keep more memory rather than strict stationarity, but it's a trade-off.
4. **Fees**: I assumed a flat 5 bps fee, but real-world slippage can be worse depending on liquidity.
5. **Single Test Window**: I only tested this on the 2024-2025 market environment. A robust strategy needs to be tested across multiple different market crashes and regimes.

---

## 18. Project Architecture Index

| File | Purpose |
|------|---------|
| `src/fetch_data.py` | Downloads data from Yahoo Finance |
| `src/data_cleaning.ipynb` | Syncs the dates across all assets |
| `src/feature_engineering.ipynb` | Builds the technical features and labels |
| `src/indicators.py` | The math for the technical indicators |
| `src/frac_diff.py` | The fractional differencing code |
| `src/triple_barrier.py` | The labeling code |
| `src/train_trade_filter.py` | Trains the XGBoost model |
| `src/backtest_engine.py` | Calculates returns and Sharpe ratios |
| `src/monte_carlo_audit.py` | Runs the random simulations |
| `src/threshold_optimizer.py` | Finds the best probability threshold |
| `src/shap_audit.py` | Checks feature importance |
