import yfinance as yf 
import pandas as pd
import json

with open(r"PINNs\data_accumulation\config.json","r") as file:
    conf=json.load(file)

data_aapl=yf.download("AAPL",start="2023-06-01",end="2026-06-01")
csv_data=data_aapl.to_csv(r"C:\Users\Admin\Desktop\HJB_MERTON_BENCHMARK\PINNs\data_accumulation\aapl_data.csv")




