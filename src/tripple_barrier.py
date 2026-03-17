import polars as pl
import numpy as np


def generate_barriers(df,k,T):
    labels = []
    opens = df['Open'].to_list()
    high = df['High'].to_list()
    low = df['Low'].to_list()
    vol_20d = df['Vol_20d'].to_list()
    for i in range(len(opens)-T-1):
        label = 0 # base case that T days passed without anything
        p0 = opens[i+1]
        sigma = vol_20d[i]
        p_upper = p0*(1+k*sigma)
        p_lower = p0*(1-k*sigma)

        for j in range(1,T+1):
            if(high[i+j] >= p_upper) and (low[i+j] <= p_lower):
                label = -1 # assuming low was first to be safe
                break
            elif(high[i+j] >= p_upper):
                label = 1 # profit
                break
            elif(low[i+j] <= p_lower):
                label = -1 # stop loss
                break
            else:
                pass
        labels.append(label)
    return labels


df1 = pl.read_parquet(r"data\processed\train\nifty.parquet")
df2 = pl.read_parquet(r"data\processed\train\gold.parquet")
df3 = pl.read_parquet(r"data\processed\train\usdinr.parquet")


with open("reports/tripple_barrier/nifty_grid_results.txt", "w",encoding="utf-8") as f:
    for k in [0.5, 0.75, 1.0, 1.5]:
        for t in [3, 5, 10]:
            labels = generate_barriers(df1, k=k, T=t)
            counts = pl.Series("TB_Label", labels).value_counts().sort("TB_Label")
            total = counts["count"].sum()
            counts = counts.with_columns((pl.col("count") / total * 100).round(1).alias("pct"))
            header = f"\nk={k}, T={t}:"
            print(header)
            print(counts)
            f.write(header + "\n")
            f.write(str(counts) + "\n")
