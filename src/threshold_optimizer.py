import polars as pl
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration
VAL_BLOB_DIR = "models/meta"
ASSETS = ["nifty", "gold", "usdinr"]
FRICTIONS = [0.0, 0.0005]  # 0 bps (Gross) and 5 bps (Net)
THRESHOLD_RANGE = np.arange(0.45, 0.61, 0.01)

pl.Config.set_tbl_width_chars(160)
pl.Config.set_fmt_str_lengths(20)
pl.Config.set_tbl_rows(100)

def calculate_sharpe(returns):
    if len(returns) < 5:
        return 0.0
    mean_ret = returns.mean()
    std_ret  = returns.std()
    if std_ret == 0 or np.isnan(std_ret):
        return 0.0
    return (mean_ret / std_ret) * np.sqrt(252)

def calculate_mdd(returns):
    #Max Drawdown calculation
    if len(returns) < 5:
        return 0.0
    cum_ret = np.cumsum(returns)
    peak = np.maximum.accumulate(cum_ret)
    drawdown = peak - cum_ret
    return np.max(drawdown) if len(drawdown) > 0 else 0.0

def plot_dual_sweep(df_0, df_5, asset):
    plt.figure(figsize=(14, 8))
    
    # Primary Axis: Sharpe Ratio
    ax1 = plt.gca()
    ax1.plot(df_0["threshold"], df_0["sharpe"], marker='o', color='#1f77b4', label='Sharpe (0 bps)', linewidth=2.5)
    ax1.plot(df_5["threshold"], df_5["sharpe"], marker='s', color='#ff7f0e', label='Sharpe (5 bps)', linewidth=2.5, linestyle='--')
    
    ax1.set_xlabel('Probability Threshold', fontsize=12)
    ax1.set_ylabel('Annualized Sharpe Ratio', fontsize=12, color='#1f77b4')
    ax1.grid(True, alpha=0.2)

    idx_pk = df_5["sharpe"].arg_max()
    pk_thr = df_5["threshold"][idx_pk]
    pk_sha = df_5["sharpe"][idx_pk]
    ax1.annotate(f"Alpha Peak\nT={pk_thr:.2f}", 
                 xy=(pk_thr, pk_sha), xytext=(pk_thr+0.02, pk_sha+0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    ax1.legend(loc='upper left')
    
    # Secondary Axis: Calmar Ratio
    ax2 = ax1.twinx()
    ax2.plot(df_5["threshold"], df_5["calmar"], marker='x', color='#2ca02c', label='Calmar (5 bps)', alpha=0.6, linewidth=1.5)
    ax2.set_ylabel('Calmar Ratio (Risk-Adjusted)', color='#2ca02c', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    
    plt.title(f'METR - {asset.upper()} Friction Audit (Gross vs. Net)', fontsize=14)
    plt.tight_layout()
    
    plot_path = f"reports/threshold_optimization/{asset}_friction_sensitivity.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def main():
    os.makedirs("reports/threshold_optimization", exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  METR - THRESHOLD OPTIMIZER")
    print(f"  Metrics: Net Sharpe, MDD, Calmar")
    print(f"{'='*70}")

    for asset in ASSETS:
        blob_path = f"{VAL_BLOB_DIR}/{asset}_val_blob.parquet"
        if not os.path.exists(blob_path):
            print(f"   [SKIP] {asset}: Missing validation blob.")
            continue
        
        blob = pl.read_parquet(blob_path)
        print(f"\n   Auditing {asset.upper()}...")
        
        # Calculate overall baseline win rate
        baseline_win_rate = (blob["Directional_Return"] > 0).mean()
        
        all_friction_data = []
        for f_val in FRICTIONS:
            sweep_data = []
            for t in THRESHOLD_RANGE:
                subset = blob.filter(pl.col("OOF_Prob") >= t)
                net_returns = subset["Directional_Return"] - f_val
                
                trade_count = len(subset)
                trade_fraction = trade_count / len(blob) if len(blob) > 0 else 0
                win_rate = (subset["Directional_Return"] > 0).mean() if trade_count > 0 else 0
                edge_gain = win_rate - baseline_win_rate
                
                sharpe = calculate_sharpe(net_returns)
                mdd = calculate_mdd(net_returns)
                calmar = sharpe / mdd if mdd > 0 else 0.0
                
                sweep_data.append({
                    "threshold": round(t, 2),
                    "sharpe": round(sharpe, 4),
                    "mdd": round(mdd, 4),
                    "calmar": round(calmar, 4),
                    "trades": trade_count,
                    "fraction": round(trade_fraction, 4),
                    "win_rate": round(win_rate, 4),
                    "edge": round(edge_gain, 4)
                })
            all_friction_data.append(pl.DataFrame(sweep_data))
            
        df_0 = all_friction_data[0]
        df_5 = all_friction_data[1]
        
        # Save Consolidated Sweep Result as TXT
        txt_path = f"reports/threshold_optimization/{asset}_sweep_audit.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"--- {asset.upper()} AUDIT on Train Data---\n")
            f.write(f"Baseline Win Rate: {baseline_win_rate:.4f}\n\n")
            
            f.write(f"--- 0 bps (Gross) ---\n")
            f.write(str(df_0) + "\n\n")
            
            f.write(f"--- 5 bps (Net) ---\n")
            f.write(str(df_5) + "\n\n")
        
        # Generate Visualization
        plot_path = plot_dual_sweep(df_0, df_5, asset)
        
        # Display Top Snapshots
        net_pk = df_5.sort("sharpe", descending=True).row(0, named=True)
        print(f"     [ALPHA PK] T={net_pk['threshold']:.2f} | Sharpe={net_pk['sharpe']:.2f} | Calmar={net_pk['calmar']:.2f} | MDD={net_pk['mdd']:.4f}")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
