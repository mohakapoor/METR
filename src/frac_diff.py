import numpy as np
import polars as pl
import numpy as np
from statsmodels.tsa.stattools import adfuller

def get_weights(d, size):
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)

def frac_diff(series, d, thresh=1e-5):
    weights = get_weights(d, len(series))
    
    # truncate small weights
    weights = weights[np.abs(weights) > thresh]
    
    out = np.zeros(len(series))
    out[:] = np.nan
    
    for i in range(len(weights), len(series)):
        out[i] = np.dot(weights[::-1], series[i-len(weights):i])
    
    return out

def evaluate_d(assets):
    for name, df in assets:
        series = np.array(df['Close'])
    
        for d in np.arange(0.1, 1.0, 0.1):
            fd_series = frac_diff(series, d)
            fd_series = fd_series[~np.isnan(fd_series)]
            if len(fd_series) < 20:
                continue
            pval = adfuller(fd_series)[1]
            print(f"d={d:.2f}, p-value={pval}, asset = {name}")




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