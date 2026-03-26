import numpy as np
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def get_weights(d, size):
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)

def frac_diff(series, d, thresh=1e-5):
    weights = get_weights(d, len(series))

    weights = weights[np.abs(weights) > thresh]
    
    out = np.zeros(len(series))
    out[:] = np.nan
    
    for i in range(len(weights), len(series)):
        out[i] = np.dot(weights[::-1], series[i-len(weights):i])
    
    return out

def best_d_value(d_values,pvals):
    if len(d_values) == len(pvals):
        scores = sorted(zip(d_values,pvals))
        for i,j in scores:
            if(j<0.05):
                return i

def evaluate_d(assets):
    for name, df in assets:
        series = np.array(df['Close'])
        d_values = np.arange(0.1,1.0,0.1)
        pvals = []
        with open(f"reports/tripple_barrier/{name}_grid_results.txt", "w", encoding="utf-8") as f:
            f.write(name.upper() + ":\n")
            for d in d_values:
                fd_series = frac_diff(series, d)
                fd_series = fd_series[~np.isnan(fd_series)]
                if len(fd_series) < 20:
                    pvals.append(np.nan)
                    continue
                try:
                    pval = adfuller(fd_series)[1]
                except:
                    pval = np.nan
                f.write(f"d={d:.2f}, p-value={pval} \n")
                print(f"d={d:.2f}, p-value={pval}, asset = {name}")
                pvals.append(pval)
            best_d = best_d_value(d_values,pvals)

        

        plt.figure()
        plt.plot(d_values, pvals, marker='o')
        plt.axhline(0.05)  # threshold line
        plt.title(f"ADF p-value vs d ({name})")
        plt.xlabel("d")
        plt.ylabel("p-value  (log)")
        plt.yscale('log')
        plt.show()
    



if __name__ == "__main__":
    df1 = pl.read_parquet(r"data\processed\train\nifty.parquet")
    df2 = pl.read_parquet(r"data\processed\train\gold.parquet")
    df3 = pl.read_parquet(r"data\processed\train\usdinr.parquet")
    assets = [
        ("nifty", df1),
        ("gold", df2),
        ("usdinr", df3),
    ]
    evaluate_d(assets=assets)