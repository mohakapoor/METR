import polars as pl
import pandas as pd
import frac_diff as frd
import triple_barrier as tb

def generate_features(asset,k,d):
    # base expressions
    c = pl.col("Close")
    o = pl.col("Open")
    h = pl.col("High")
    l = pl.col("Low")

    # Macd Calc
    ema_12 = c.ewm_mean(span=12,adjust=False)
    ema_26 = c.ewm_mean(span=26,adjust=False)
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm_mean(span=9,adjust=False)


    # Rsi Calc
    delta = c.diff()
    up = delta.clip(lower_bound =0)
    down = delta.clip(upper_bound=0).abs()
    roll_up = up.ewm_mean(com=13,ignore_nulls = True)
    roll_down = down.ewm_mean(com=13,ignore_nulls = True)
    rs = roll_up/roll_down
    rsi = 100.0 -(100.0/(1.0+rs))

    # BB Calc
    bb_mean = c.rolling_mean(window_size=20)
    bb_std = c.rolling_std(window_size=20)
    bb_up = bb_mean + (2*bb_std)
    bb_low = bb_mean - (2*bb_std)

    # True Range Calc
    
    prev_close = c.shift(1)
    tr1 = h -l
    tr2 = (h-prev_close).abs()
    tr3 = (l-prev_close).abs()
    tr = pl.max_horizontal([tr1,tr2,tr3])
    atr = tr.ewm_mean(span=14,adjust=False)


    asset = (
        asset.lazy()
        .with_columns(
            #Returns
            c.pct_change().alias("Ret_1d"),
            c.pct_change(n=3).alias("Ret_3d"),
            c.pct_change(n=5).alias("Ret_5d"),
            c.pct_change(n=20).alias("Ret_20d"),
        )
        .with_columns(
            #Volatility
            pl.col("Ret_1d").rolling_std(window_size=5).alias("Vol_5d"),
            pl.col("Ret_1d").rolling_std(window_size=20).alias("Vol_20d")
        )
        .collect()
    )
    # frac diff 
    close_np = asset["Close"].to_numpy()
    fd = frd.frac_diff(close_np,d,thresh=1e-4)

    # Labels
    labels, returns = tb.generate_barriers(asset, k=k, T=3) #T=3 is fixed
    asset = asset.with_columns([
        pl.Series("FD_Close", fd),
        pl.Series("Label", labels),
    ])
    q = (
        asset.lazy()
        .with_columns(
            #MAs
            c.rolling_mean(window_size=5).alias("MA_5d"),
            c.rolling_mean(window_size=20).alias("MA_20d"),
            
            macd_line.alias("Macd_Line"),
            signal_line.alias("Signal_Line"),

            #FracDiff Lag
            pl.col('FD_Close').shift(1).alias('FD_Close_Lag1')
        )
        .with_columns(
            
            (pl.col("Macd_Line")-pl.col("Signal_Line")).alias("MACD_Hist"),
            rsi.alias("RSI"),
            ((c/c.shift(10)-1)*100).alias("ROC_10"),

            ((c-c.shift(20))/(c.shift(20))).alias("Trend_Strength"),
            ((o-c.shift(1))/(c.shift(1))).alias("Gap"), 

            pl.when((bb_up-bb_low)!=0)
            .then((c-bb_low)/(bb_up-bb_low))
            .otherwise(0.5)
            .alias("BB_Pct"),
            (atr/c).alias("ATR_Pct"),
        )
        .with_columns(
            pl.when(pl.col("Vol_20d") != 0)
            .then(pl.col("Vol_5d") / pl.col("Vol_20d"))
            .otherwise(1)
            .alias("Vol_Ratio"),

            (pl.col("MA_5d")/pl.col("MA_20d")).alias("MA_Ratio"),
            (c/pl.col("MA_20d")).alias("Price_vs_MA20"),

            pl.when((h-l)!=0)
            .then((c-l)/(h-l))
            .otherwise(0.5)
            .alias("Close_Pos_Range"),

            ((c-o)/o).alias("Intraday_Return"),
            pl.when((h.shift(1)-l.shift(1)) != 0)
            .then((h-l)/(h.shift(1)-l.shift(1)))
            .otherwise(1)
            .alias("Range_Expansion"),
        )
        .drop(["MA_5d", "MA_20d", 'Macd_Line','Signal_Line',])
    )

    return q.collect()