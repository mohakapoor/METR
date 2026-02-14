import polars as pl
import numpy as np  
import yfinance as yf   

tickers = {"Nifty50":"^NSEI","Gold":"GOLDBEES.NS","USDINR":"USDINR=X"}

for name,tick in tickers.items():
    # datH = yf.download(tick, start='2024-3-01', end='2026-01-01',interval='1h')
    # datH.to_parquet(f'data/raw/{name}_1h.parquet')
    # print(f'{name} hourly data collected and saved')
    datD = yf.download(tick, start='2013-01-01', end='2026-01-01',interval='1d') 
    datD.to_parquet(f'data/raw/{name}_1d.parquet')
    print(f'{name} daily data collected and saved')
