import polars as pl
import numpy as np  
import yfinance as yf   

tickers = ['^NSEI','GOLDBEES.NS','USDINR=X']

for tick in tickers:
    datH = yf.download(tick, start='2024-3-01', end='2026-01-01',interval='1h')
    datH.to_parquet(f'data/raw/{tick}_1h.parquet')
    print(f'{tick} hourly data collected and saved')
    datD = yf.download(tick, start='2014-01-01', end='2026-01-01',interval='1d') 
    datD.to_parquet(f'data/raw/{tick}_1d.parquet')
    print(f'{tick} daily data collected and saved')
