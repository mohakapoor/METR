import polars as pl
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration
VAL_BLOB_DIR = "models/meta"
ASSETS = ["nifty", "gold", "usdinr"]
FRICTIONS = [0.0, 0.0005]  # 0 bps (Gross) and 5 bps (Net)
THRESHOLD_RANGE = np.arange(0.45, 0.61, 0.01)

def calculate_sharpe(returns):
    if len(returns) < 5:
        return 0.0
    mean_ret = returns.mean()
    std_ret  = returns.std()
    if std_ret == 0 or np.isnan(std_ret):
        return 0.0
    return (mean_ret / std_ret) * np.sqrt(252)

def plot_dual_sweep(df_0, df_5, asset):
    plt.figure(figsize=(12, 7))
    
    # Primary Axis: Sharpe Ratio
    ax1 = plt.gca()
    ax1.plot(df_0["threshold"], df_0["sharpe"], marker='o', color='#1f77b4', label='Sharpe (0 bps)', linewidth=2.5)
    ax1.plot(df_5["threshold"], df_5["sharpe"], marker='s', color='#ff7f0e', label='Sharpe (5 bps)', linewidth=2.5, linestyle='--')
    
    ax1.set_xlabel('Probability Threshold', fontsize=12)
    ax1.set_ylabel('Annualized Sharpe Ratio', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Secondary Axis: Trade Count (using the 0 bps count as reference)
    ax2 = ax1.twinx()
    ax2.bar(df_0["threshold"], df_0["trades"], alpha=0.1, color='gray', label='Trade Count', width=0.006)
    ax2.set_ylabel('Number of Trades', color='gray', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='gray')
    
    plt.title(f'METR - {asset.upper()} Friction Sensitivity Audit (Gross vs. Net)', fontsize=14)
    plt.tight_layout()
    
    plot_path = f"reports/threshold_optimization/{asset}_friction_sensitivity.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def main():
    os.makedirs("reports/threshold_optimization", exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  METR THRESHOLD OPTIMIZER — Integrated Friction Sensitivity Audit")
    print(f"  Visualizing: Gross Alpha (0 bps) vs. Net Alpha (5 bps)")
    print(f"  Logging results to: reports/threshold_optimization/")
    print(f"{'='*70}")

    for asset in ASSETS:
        blob_path = f"{VAL_BLOB_DIR}/{asset}_val_blob.parquet"
        if not os.path.exists(blob_path):
            print(f"   [SKIP] {asset}: Validation blob not found.")
            continue
        
        blob = pl.read_parquet(blob_path)
        print(f"\n   Auditing {asset.upper()} ({len(blob)} OOF samples)...")
        
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
                
                sweep_data.append({
                    "threshold": round(t, 2),
                    "sharpe": round(sharpe, 4),
                    "trades": trade_count,
                    "fraction": round(trade_fraction, 4),
                    "win_rate": round(win_rate, 4),
                    "edge": round(edge_gain, 4),
                    "baseline": round(baseline_win_rate, 4)
                })
            all_friction_data.append(pl.DataFrame(sweep_data))
            
        df_0 = all_friction_data[0]
        df_5 = all_friction_data[1]
        
        # Save Consolidated Sweep Result as TXT
        txt_path = f"reports/threshold_optimization/{asset}_sweep_audit.txt"
        with pl.Config(tbl_rows=100):
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"--- {asset.upper()} FRICTION SENSITIVITY AUDIT ---\n")
                f.write(f"Baseline Win Rate (Unfiltered): {baseline_win_rate:.4f}\n\n")
                
                f.write(f"--- PASS 1: GROSS ALPHA (0 bps friction) ---\n")
                f.write(str(df_0) + "\n\n")
                
                f.write(f"--- PASS 2: NET ALPHA (5 bps friction) ---\n")
                f.write(str(df_5) + "\n\n")
        
        # Generate Visualization
        plot_path = plot_dual_sweep(df_0, df_5, asset)
        
        # Display Top Snapshots
        print(f"     [GROSS PK] T={df_0.sort('sharpe', descending=True)['threshold'][0]:.2f} | Sharpe={df_0.sort('sharpe', descending=True)['sharpe'][0]:.2f}")
        print(f"     [NET PK]   T={df_5.sort('sharpe', descending=True)['threshold'][0]:.2f} | Sharpe={df_5.sort('sharpe', descending=True)['sharpe'][0]:.2f}")

    print(f"\n{'='*70}")
    print(f"  INTEGRATED AUDIT COMPLETE. Reports generated in reports/threshold_optimization/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
