import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import black_litterman

def black_litterman_optimization(tickers_views, market_caps={}, period="10y", risk_aversion_scale=0.5):
    """
    Performs Black-Litterman portfolio optimization given a dictionary of tickers and expected gains.

    Args:
        tickers_views (dict): Dictionary of stock tickers and expected gains (absolute views).
        market_caps (dict): Dictionary of stock tickers and market capitalizations.
        period (str): Period for historical data (default is "10y").
        risk_aversion_scale (float): Scaling factor for market-implied risk aversion (default is 0.5).

    Returns:
        dict: Optimized portfolio weights
        dict: Portfolio performance metrics (expected return, volatility, Sharpe ratio)
    """

    def get_market_caps(tickers):
        """
        Generates market caps dict from tickers list, only if no market caps are given
        
        Args:
            tickers (list): List of tickers, can be found with tickers_views(dict).keys()

        Returns:
            dict: Tickers and market caps
        """
        real_market_caps = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            real_market_caps[ticker] = stock.info.get("marketCap", 1e9)  # Default 1B if missing
        return real_market_caps

    # Extract tickers from the provided dictionary
    assets = list(tickers_views.keys())

    #Download historical data
    df = yf.download(assets, period=period)['Close'].ffill()

    #Compute log returns and convert to simple returns
    log_returns = np.log(df / df.shift(1)).dropna()
    simple_returns = np.exp(log_returns) - 1 

    #Compute covariance matrix
    cov_matrix = np.cov(simple_returns, rowvar=False)
    cov_matrix_df = pd.DataFrame(cov_matrix, index=assets, columns=assets)

    #Compute market capitalization weights, if none given, most recent caps found
    if not market_caps:
        market_caps = get_market_caps(tickers_views.keys())
    market_weights = {ticker: cap / sum(market_caps.values()) for ticker, cap in market_caps.items()}

    #Compute market-implied prior returns (pi)
    delta = black_litterman.market_implied_risk_aversion(df) * risk_aversion_scale  # Scale down delta
    pi = black_litterman.market_implied_prior_returns(market_weights, delta, cov_matrix_df)

    #Run Black-Litterman Model
    bl = BlackLittermanModel(cov_matrix_df, pi=pi, tickers=assets, absolute_views=tickers_views)
    bl_returns = bl.bl_returns() * 252  # Annualize returns
    bl_cov_matrix = bl.bl_cov()

    #Optimize Portfolio
    ef = EfficientFrontier(bl_returns, bl_cov_matrix)
    weights = ef.max_sharpe()

    #Compute Portfolio Performance
    expected_return, volatility, sharpe_ratio = ef.portfolio_performance(verbose=False)

    # Format Results
    opt_weights = dict(ef.clean_weights())
    exp_ret = float(round(expected_return,2))
    vol = float(round(volatility,4))
    sharpe = float(round(sharpe_ratio,2))
    

    #Return results as dictionaries
    return {
            "Optimized Weights": opt_weights,
            "Portfolio Performance": {
                "Expected Annual Return": exp_ret,
                "Volatility": vol,
                "Sharpe Ratio": sharpe
            }}
    

# Define stock tickers and expected returns
tickers_views = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.15}

# Define market capitalizations for the assets, not used here
market_caps = {
    "AAPL": 2.8e12,
    "BBY": 0.02e12,
    "BAC": 0.34e12,
    "SBUX": 0.09e12,
    "T": 0.14e12
}

result = black_litterman_optimization(tickers_views, period="10y")
print("Portfolio Performance:", result["portfolio_performance"])






