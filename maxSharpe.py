import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import yfinance as yf
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


def get_historical_prices(assets,period):
    valid_tickers = [ticker for ticker in assets if not yf.Ticker(ticker).history(period="1y").empty]
    return yf.download(valid_tickers, period=period)['Close'].ffill()

def max_sharpe_optimization(df,ticker_list):
    formatted_tickers_views = format_tickers(ticker_list)
    assets = ticker_list
    
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Optimize for maximal Sharpe ratios
    ef = EfficientFrontier(mu, S)
    
    #Prints weights, dont bother supressing
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    return cleaned_weights

def portfolio_allocation(df,cleaned_weights):
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=10000)

    allocation, leftover = da.greedy_portfolio()
    return ef.portfolio_performance(verbose=False)
    
def format_tickers(tickers):
    return {ticker: 0 for ticker in tickers}