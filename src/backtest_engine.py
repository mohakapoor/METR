import polars as pl
import numpy as np
import os
import matplotlib.pyplot as plt

ASSET_CONFIG = {
    "nifty": {"threshold": 0.49, "label": "Nifty"},
    "gold":  {"threshold": 0.52, "label": "Gold"},
    "usdinr": {"threshold": 0.50, "label": "USD/INR"}
}
FRICTION = 0.0005  # 5 bps
DATA_DIR = "models/meta"
REPORT_DIR = "reports/backtest"

def calculate_metrics(returns):
    if len(returns) < 5:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # 1. Annualized Sharpe
    n_years = len(returns) / 252
    mean_ret = returns.mean()
    std_ret  = returns.std()
    account_sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
    
    # 2. Annualized Sharpe (Trade-Frequency)
    trade_returns = returns[returns != 0]
    n_trades = len(trade_returns)
    trades_per_year = n_trades / n_years if n_years > 0 else 0.0
    
    if n_trades > 5 and trade_returns.std() > 0:
        signal_sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(trades_per_year)
    else:
        signal_sharpe = 0.0
    
    # 3. Compounded Equity and True MDD
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    mdd = np.abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    
    # 4. Calmar Ratio (Annualized Return / MDD)
    total_comp_ret = equity[-1] - 1
    ann_return = (equity[-1] ** (1/n_years)) - 1 if n_years > 0 and equity[-1] > 0 else 0.0
    calmar = ann_return / mdd if mdd > 0 else 0.0
    
    return total_comp_ret, account_sharpe, signal_sharpe, mdd, calmar

def generate_asset_plots(df, asset_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. Equity Curve
    ax1.plot(df["Date"], df["cum_ret"], label=f"{asset_name.upper()} Net Equity", color="#1f77b4", linewidth=2.5)
    ax1.set_title(f"METR - OOS Performance: {asset_name.upper()} (2024-2025)", fontsize=14)
    ax1.set_ylabel("Cumulative Return (Compounded)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.15)

    # 2. Underwater Plot
    equity = df["cum_ret"] + 1  # Reconstruct as Equity Line (Starting at 1)
    peak = equity.cum_max()
    drawdown = (equity - peak) / peak
    ax2.fill_between(df["Date"], drawdown, 0, color="#d62728", alpha=0.3)
    ax2.plot(df["Date"], drawdown, color="#d62728", linewidth=1)
    ax2.set_ylabel("Drawdown", fontsize=12)
    ax2.grid(True, alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/{asset_name}_report.png")
    plt.close()

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  METR PER-ASSET BACKTEST ")
    print(f"{'='*70}")

    log_path = f"{REPORT_DIR}/log.txt"
    with open(log_path, "w", encoding="utf-8") as f_log:
        f_log.write(f"--- METR PER-ASSET BACKTEST LOG ---\n")
        f_log.write(f"Period: 2024 - 2025 (Out of Sample)\n\n")

        for asset, config in ASSET_CONFIG.items():
            path = f"{DATA_DIR}/{asset}_test_blob.parquet"
            if not os.path.exists(path):
                print(f"   [SKIP] {asset} test blob missing.")
                continue
            
            # Load & Process
            df = pl.read_parquet(path).sort("Date").filter(pl.col("Directional_Return").is_not_null())
            
            # Apply Production Threshold & Friction
            df = df.with_columns([
                pl.when(pl.col("OOF_Prob") >= config["threshold"])
                .then(pl.col("Directional_Return") - FRICTION)
                .otherwise(0.0)
                .alias("net_ret")
            ])
            
            # Reconstruct Cumulative (Compounded)
            df = df.with_columns([
                (pl.col("net_ret") + 1).cum_prod().alias("equity")
            ])
            df = df.with_columns([
                (pl.col("equity") - 1).alias("cum_ret")
            ])
            
            # Forensic Metrics
            total_ret, acc_sharpe, sig_sharpe, mdd, calmar = calculate_metrics(df["net_ret"].to_numpy())
            
            # Logging
            f_log.write(f"[{asset.upper()}] T={config['threshold']}\n")
            f_log.write(f"   Total Ret: {total_ret:.4f} | Acc Sharpe: {acc_sharpe:.2f} | Sig Sharpe: {sig_sharpe:.2f} | MDD: {mdd:.4f} | Calmar: {calmar:.2f}\n\n")
            
            # Reporting
            generate_asset_plots(df, asset)
            
            print(f"   Processed {asset.upper()}: Acc Sharpe {acc_sharpe:.2f} | Sig Sharpe {sig_sharpe:.2f} | Calmar {calmar:.2f}")

    print(f"\n   Reports generated in {REPORT_DIR}/")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()