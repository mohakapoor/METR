# METR — Market Exposure Timing vs Randomness

## Overview
**METR** is a controlled empirical study designed to investigate whether structured machine learning models can outperform pure randomness in short-term asset allocation. The project focuses on a fixed 3-day holding period for three distinct asset classes: Equity (NIFTY), Gold, and Currency (USDINR).

> **Current Status:** The project is in the **Data Processing** phase. Data ingestion and feature engineering pipelines are implemented.

## Project Structure
```
METR/
├── data/
│   ├── raw/            # Raw parquet files from Yahoo Finance
│   └── processed/      # Cleaned data with features (train/test splits)
├── docs/               # Project documentation and plans
├── src/
│   ├── fetch_data.py   # Script to download historical data
│   └── data_cleaning.ipynb # Feature engineering and preprocessing
├── config.yaml         # Configuration configurations
└── README.md           # This file
```

## Features & Functionality

### 1. Data Ingestion
*   **Source:** Yahoo Finance (`yfinance`).
*   **Assets:** Nifty50 (`^NSEI`), Gold (`GOLDBEES.NS`), USDINR (`USDINR=X`).
*   **Automation:** `src/fetch_data.py` automates the download of daily and hourly data, saving them as Parquet files in `data/raw/`.

### 2. Data Processing & Feature Engineering
*   **Cleaning:** Filters out invalid dates and handles missing values.
*   **Feature Generation:**
    *   **Returns:** 1-day, 3-day, 5-day, and 20-day percentage changes.
    *   **Volatility:** Rolling standard deviations (5d, 10d, 20d) and volatility ratios.
    *   **Trend:** Moving averages (5d, 20d) and their ratios.
    *   **Intraday:** Close-to-Open and High-Low range dynamics.
*   **Labeling:** Generates forward-looking labels based on 3-day returns for supervised learning.
*   **Output:** Processes data into training and testing sets, saved in `data/processed/`.

### 3. Data Exploration & Insights
*   **Asset Correlation:**
    *   **Nifty vs. Gold:** Low correlation, indicating potential diversification benefits.
    *   **USDINR:** Acts as a regime filter, often spiking during stress periods (negative correlation with Nifty).
*   **Stationarity:** Checks confirm that raw prices are non-stationary, but percentage returns (`Ret_1d`, `Ret_3d`) are stationary, validating their use as model features.
*   **Distribution:** Daily returns exhibit "fat tails" (kurtosis > 3), justifying the use of robust models like XGBoost over linear regression.

## Getting Started

### Prerequisites
*   Python 3.10+
*   `polars`, `yfinance`, `pyyaml`, `numpy`, `matplotlib`

### Usage
1.  **Download Data:**
    ```bash
    python src/fetch_data.py
    ```
2.  **Process Data:**
    Open and run `src/data_cleaning.ipynb` to generate features and save processed datasets.
