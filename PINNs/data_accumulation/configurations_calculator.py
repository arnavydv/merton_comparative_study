import pandas as pd
import numpy as np
import json

with open(r"C:\Users\Admin\Desktop\HJB_MERTON_BENCHMARK\PINNs\data_accumulation\config.json", "r") as file:
    confi=json.load(file)
data=pd.read_csv("PINNs/data_accumulation/aapl_data.csv")
data=data.dropna()
data=data[1:]
data=pd.to_numeric(data["Close"])
log_returns = np.log(data/data.shift(1))
log_returns_mean=log_returns.mean()
mu=log_returns_mean*confi["Time"]
sigma=log_returns.std()
sigma_annual=sigma*(np.sqrt(confi["Time"]))

