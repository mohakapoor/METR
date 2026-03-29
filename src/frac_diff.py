import numpy as np
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import yaml
from statsmodels.tsa.stattools import adfuller

def get_weights(d,thresh):
    w, k = [1.0], 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w)

def frac_diff(series, d, thresh):
    weights = get_weights(d,thresh)
    
    out = np.full(len(series),np.nan)
    
    for i in range(len(weights)-1, len(series)):
        out[i] = np.dot(weights[::-1], series[i-len(weights)+1:i+1])
    
    return out

def best_d_value(d_values,pvals):
    if len(d_values) == len(pvals):
        scores = sorted(zip(d_values,pvals))
        for i,j in scores:
            if(j<0.05):
                return i

def evaluate_d(assets,thresh):
    import os
    os.makedirs("reports/frac_diff", exist_ok=True)
    dvals = {}
    for name, df in assets:
        series = np.array(df['Close'])
        d_values = np.arange(0.1,1.0,0.05)
        corrs = []
        pvals = []
        with open(f"reports/frac_diff/{name}_d_values.txt", "w", encoding="utf-8") as f:
            f.write(name.upper() + ":\n")
            for d in d_values:
                fd_series = frac_diff(series, d,thresh)
                valid_idx = ~np.isnan(fd_series)
                fd_series = fd_series[valid_idx]
                if len(fd_series) < 20:
                    pvals.append(np.nan)
                    continue
                try:
                    pval = adfuller(fd_series)[1]
                except:
                    pval = np.nan
                corr = np.corrcoef(fd_series,series[valid_idx])[0,1]
                f.write(f"d={d:.2f}, p-value={pval}, corr = {corr} \n")
                print(f"d={d:.2f}, p-value={pval}, asset = {name}, corr = {corr}")
                corrs.append(corr)
                pvals.append(pval)
            
            best_d = best_d_value(d_values,pvals)
            dvals[name] = best_d
            f.write(f"best d_value = {best_d}")
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Primary Axis: ADF p-value (Log Scale)
        ax1.set_xlabel('d')
        ax1.set_ylabel('ADF p-value (log)', color='tab:blue')
        ax1.plot(d_values, pvals, marker='o', color='tab:blue', label='ADF p-value')
        ax1.axhline(0.05, color='red', linestyle='--', label='95% Confidence (0.05)')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, which="both", ls="-", alpha=0.3)

        # Secondary Axis: Correlation (Memory)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Correlation (Memory Preservation)', color='tab:green')
        ax2.plot(d_values, corrs, marker='x', linestyle='--', color='tab:green', label='Correlation')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.set_ylim(0, 1.1)

        plt.title(f"FracDiff Trade-off: Stationarity vs Memory ({name.upper()})")
        fig.tight_layout()
        
        # Combined Legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='center left')

        plt.savefig(f"reports/frac_diff/{name}_d_value_analysis.png")
        plt.close() # Close to free memory
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config['Fractional_Differentiation'] = {k: round(float(v), 2) if v is not None else None for k, v in dvals.items()}
    
    with open("config.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    print("\nUpdated config.yaml with best d-values.")
    



if __name__ == "__main__":
    df1 = pl.read_parquet(r"data\processed\train\nifty.parquet")
    df2 = pl.read_parquet(r"data\processed\train\gold.parquet")
    df3 = pl.read_parquet(r"data\processed\train\usdinr.parquet")
    assets = [
        ("nifty", df1),
        ("gold", df2),
        ("usdinr", df3),
    ]
    evaluate_d(assets=assets,thresh=1e-4)
    # def get_L(d, thresh=1e-3):
    #     w, k = [1.0], 1
    #     while True:
    #         w_k = -w[-1] * (d - k + 1) / k
    #         if abs(w_k) < thresh:
    #             break
    #         w.append(w_k)
    #         k += 1
    #     return len(w)
    # print(f"threshold is: {1e-3}")
    # for d in np.arange(0.1,1.0,0.05):
    #     print(f"d={d:.2f} → L={get_L(d)}")