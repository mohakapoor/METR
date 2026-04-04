import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.stats import binomtest, chi2
ASSETS = {
    "gold": {"path": "models/meta/gold_test_blob.parquet", "t": 0.52},
    "nifty": {"path": "models/meta/nifty_test_blob.parquet", "t": 0.49},
    "usdinr": {"path": "models/meta/usdinr_test_blob.parquet", "t": 0.50}
}
FRICTION = 0.0005
ITERATIONS = 10000
SEED = 42

def calculate_sharpe(returns, trades_per_year):
    if len(returns) < 5 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(trades_per_year)

def run_monte_carlo(asset_name, config):
    path = config["path"]
    t = config["t"]
    
    if not os.path.exists(path):
        print(f"   [SKIP] {asset_name} missing.")
        return

    df = pl.read_parquet(path).sort("Date")
    
    # 0. Temporal Basis
    n_years = (df["Date"].max() - df["Date"].min()).days / 365.25
    if n_years <= 0: n_years = 1.0
    
    # 1. Universe: All Signal Days
    df_signals = df.filter(pl.col("Directional_Return") != 0)
    signal_returns = df_signals["Directional_Return"].to_numpy()
    n_signal_days = len(signal_returns)
    baseline_tpy = n_signal_days / n_years 
    
    # 2. Actual Strategy: Model Selection
    model_returns = df_signals.filter(pl.col("OOF_Prob") >= t)["Directional_Return"].to_numpy() - FRICTION
    n_model_trades = len(model_returns)
    model_tpy = n_model_trades / n_years
    
    actual_sharpe = calculate_sharpe(model_returns, model_tpy)
    win_rate = (model_returns > 0).sum() / n_model_trades if n_model_trades > 0 else 0.0
    
    # 3. Baseline: Take Every Signal
    baseline_returns = signal_returns - FRICTION
    baseline_sharpe = calculate_sharpe(baseline_returns, baseline_tpy)


    # 4. Monte Carlo (Random Selection)
    np.random.seed(SEED)
    sim_sharpes = []
    for _ in range(ITERATIONS):
        random_selection = np.random.choice(signal_returns, size=n_model_trades, replace=False) - FRICTION
        sim_sharpes.append(calculate_sharpe(random_selection, model_tpy))

    sim_sharpes = np.array(sim_sharpes)
    p_mc = np.sum(sim_sharpes >= actual_sharpe) / ITERATIONS

    # 5. Academic Validation Suite
    # A. Sharpe Ratio T-test (Mean Return vs 0)
    t_stat, p_ttest = stats.ttest_1samp(model_returns, 0) if n_model_trades > 1 else (0, 1)

    # B. Binomial Test (Win Rate vs 50%)
    wins = (model_returns > 0).sum()
    binom_res = binomtest(wins, n=n_model_trades, p=0.5, alternative='greater') if n_model_trades > 0 else None
    p_binom = binom_res.pvalue if binom_res else 1.0

    # D. Kupiec Test (Win Rate Reliability)
    p_null = 0.5
    if n_model_trades > 0 and 0 < wins < n_model_trades:
        LR = -2 * (wins * np.log(p_null) + (n_model_trades-wins) * np.log(1-p_null) - 
                   wins * np.log(wins/n_model_trades) - (n_model_trades-wins) * np.log(1-wins/n_model_trades))
        p_kupiec = 1 - chi2.cdf(LR, df=1)
    else:
        p_kupiec = 1.0

    # 6. Interpretability Table
    table_str = [
        f"\n   {asset_name.upper()}:",
        f"      Trades: {n_model_trades} | Win Rate: {win_rate:.1%} | Model Sharpe: {actual_sharpe:.2f} | Baseline: {baseline_sharpe:.2f}",
        f"      ─────────────────────────────────────────────────────",
        f"      Test                    Statistic    P-value   Result",
        f"      ─────────────────────────────────────────────────────",
        f"      Monte Carlo (10,000)    Target {actual_sharpe:>4.2f}  {p_mc:>7.4f}   {'Y' if p_mc < 0.05 else 'N'}",
        f"      T-test (mean return)    t={t_stat:>5.2f}     {p_ttest:>7.4f}   {'Y' if p_ttest < 0.05 else 'N'}",
        f"      Binomial (win rate)     W={wins:>2}/{n_model_trades:<2}      {p_binom:>7.4f}   {'Y' if p_binom < 0.05 else 'N'}",
        f"      Kupiec (reliability)    LR Calc      {p_kupiec:>7.4f}   {'Y' if p_kupiec < 0.05 else 'N'}",
        f"      ─────────────────────────────────────────────────────\n"
    ]
    summary_text = "\n".join(table_str)
    print(summary_text)

    os.makedirs("reports/backtest", exist_ok=True)
    with open("reports/backtest/stats_summary.txt", "a", encoding="utf-8") as f:
        f.write(summary_text)

    # 7. Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(sim_sharpes, bins=50, color="#1f77b4", alpha=0.5, label="Random Picker Distribution")
    plt.axvline(actual_sharpe, color="#d62728", linestyle="--", linewidth=2.5, label=f"METR Model ({actual_sharpe:.2f})")
    plt.axvline(baseline_sharpe, color="green", linestyle="--", linewidth=2, label=f"Signal-Only Baseline ({baseline_sharpe:.2f})")
    
    plt.title(f"Hypothesis Testing: {asset_name.upper()} (2024-2025)", fontsize=13)
    plt.xlabel("Annualized Account Sharpe Ratio", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    
    plt.text(0.05, 0.95, f"MC P-Value: {p_mc:.4f}", 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.1)
    plt.savefig(f"reports/backtest/{asset_name}_monte_carlo.png")
    plt.close()
    return p_mc

def main():
    summary_file = "reports/backtest/stats_summary.txt"
    if os.path.exists(summary_file):
        os.remove(summary_file)

    print(f"\n METR HYPOTHESIS TESTING: ")
    for asset, config in ASSETS.items():
        run_monte_carlo(asset, config)

if __name__ == "__main__":
    main()
