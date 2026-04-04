import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os

# Forensic Settings
ASSETS = {
    "gold": {"path": "models/meta/gold_test_blob.parquet", "t": 0.52},
    # "nifty": {"path": "models/meta/nifty_test_blob.parquet", "t": 0.49},
    # "usdinr": {"path": "models/meta/usdinr_test_blob.parquet", "t": 0.50}
}
FRICTION = 0.0005
ITERATIONS = 10000
SEED = 42

def calculate_sharpe(returns):
    if len(returns) < 5 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(252)

def run_monte_carlo(asset_name, config):
    path = config["path"]
    t = config["t"]
    
    if not os.path.exists(path):
        print(f"   [SKIP] {asset_name} missing.")
        return

    df = pl.read_parquet(path).sort("Date")
    
    # 1. Universe: All Signal Days (The pool our model chooses from)
    signal_returns = df["Directional_Return"].to_numpy()
    n_days = len(signal_returns)
    
    # 2. Actual Strategy: Model Selection
    actual_returns = df.with_columns([
        pl.when(pl.col("OOF_Prob") >= t)
        .then(pl.col("Directional_Return") - FRICTION)
        .otherwise(0.0)
        .alias("net_ret")
    ])["net_ret"].to_numpy()
    
    actual_sharpe = calculate_sharpe(actual_returns)
    n_trades = int((df["OOF_Prob"] >= t).sum())
    
    # 3. Baseline: Take Every Signal (Signal-Only)
    baseline_returns = signal_returns - FRICTION
    # Pad with zeros if necessary (though signal_returns are already filtered for signals != 0)
    baseline_sharpe = calculate_sharpe(baseline_returns)

    print(f"\n   AUDITING {asset_name.upper()}:")
    print(f"      Trades: {n_trades} | Model Sharpe: {actual_sharpe:.2f} | Baseline Sharpe: {baseline_sharpe:.2f}")

    # 4. Simulation: Randomly select n_trades from the opportunity universe
    np.random.seed(SEED)
    sim_sharpes = []
    
    for _ in range(ITERATIONS):
        # Pick n_trades random indices
        indices = np.random.choice(n_days, size=n_trades, replace=False)
        random_selection = signal_returns[indices] - FRICTION
        
        # Pad with zeros to maintain temporal length (n_days) for correct annualization
        full_returns = np.zeros(n_days)
        full_returns[indices] = random_selection
        sim_sharpes.append(calculate_sharpe(full_returns))

    sim_sharpes = np.array(sim_sharpes)
    p_value = np.sum(sim_sharpes >= actual_sharpe) / ITERATIONS
    
    # 5. Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(sim_sharpes, bins=50, color="#1f77b4", alpha=0.5, label="Random Selection Distribution")
    plt.axvline(actual_sharpe, color="#d62728", linestyle="--", linewidth=2.5, label=f"METR Model ({actual_sharpe:.2f})")
    plt.axvline(baseline_sharpe, color="green", linestyle="--", linewidth=2, label=f"Signal-Only Baseline ({baseline_sharpe:.2f})")
    
    plt.title(f"Random Selection Audit: {asset_name.upper()} (2024-2025)", fontsize=13)
    plt.xlabel("Annualized Account Sharpe Ratio", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    
    verdict = "SKILL-BASED" if p_value < 0.05 else "LUCK / NON-SIGNIFICANT"
    plt.text(0.05, 0.95, f"P-Value: {p_value:.4f}\n{verdict}", 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.1)
    
    os.makedirs("reports/backtest", exist_ok=True)
    plt.savefig(f"reports/backtest/{asset_name}_monte_carlo.png")
    plt.close()
    
    return p_value

def main():
    print(f"\n{'='*70}")
    print(f"  METR MONTE CARLO AUDIT - RANDOM SELECTION (PHASE 12)  ")
    print(f"{'='*70}")

    for asset, config in ASSETS.items():
        p_val = run_monte_carlo(asset, config)
        if p_val is not None:
            print(f"   >>> {asset.upper()} P-Value: {p_val:.4f}")

    print(f"\n   Reports generated in reports/backtest/")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
