import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import yfinance as yf
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Read in price data

tickers_views_cg = {
    "PG": 0, "KO": 0, "PEP": 0, "JNJ": 0, "CL": 0, "UL": 0, 
    "GIS": 0, "KMB": 0, "MDLZ": 0, "COLM": 0, "EL": 0, "MCD": 0, 
    "PM": 0, "HSY": 0, "CAG": 0, "LRLCF": 0, "KHC": 0, "OR": 0,
    "DGE": 0, "IMB": 0, "RB": 0, "BN": 0, "7203.T": 0, 
    "005930.KS": 0, "066570.KS": 0, "CX": 0, "BABA": 0, "AMZN": 0, "TGT": 0, "WMT": 0, "COST": 0, "TJX": 0, "LOW": 0, 
    "NKE": 0, "SBUX": 0, "K": 0}

assets = list(tickers_views_cg.keys())
df = yf.download(assets, period='10y')['Close'].ffill()
# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximal Sharpe ratios
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print(cleaned_weights)

latest_prices = get_latest_prices(df)
da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=10000)

allocation, leftover = da.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))

ef.portfolio_performance(verbose=True)