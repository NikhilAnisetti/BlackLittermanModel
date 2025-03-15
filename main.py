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
    


# Example: SPLIT BY DIFFERENT INDUSTRIES

"""

# ENERGY

tickers_views_energy = {"XOM": 0, "CVX": 0, "COP": 0, "HES": 0, "OXY": 0, "EOG": 0, 
    "MPC": 0, "PSX": 0, "VLO": 0, "ENB": 0, "SU": 0, "CNQ": 0,
    "CVE": 0, "IMO": 0, "BP": 0, "EQNR": 0, "IBE": 0, "WMB": 0, "OKE": 0, "ET": 0,
    "HOU": 0, "CQP": 0, "BKR": 0, "TRGP": 0, "TGTX": 0, "AR": 0, "VLO": 0, "KMI": 0, "PPL": 0, "DUK": 0, "XEL": 0}



energy = black_litterman_optimization(tickers_views_energy, period="10y")
print(energy)



"""






# CONSUMER GOODS


tickers_views_cg = {
    "PG": 0, "KO": 0, "PEP": 0, "JNJ": 0, "CL": 0, "UL": 0, 
    "GIS": 0, "KMB": 0, "MDLZ": 0, "COLM": 0, "EL": 0, "MCD": 0, 
    "PM": 0, "HSY": 0, "CAG": 0, "LRLCF": 0, "KHC": 0, "OR": 0,
    "DGE": 0, "IMB": 0, "RB": 0, "BN": 0, "7203.T": 0, 
    "005930.KS": 0, "066570.KS": 0, "CX": 0, "BABA": 0, "AMZN": 0, "TGT": 0, "WMT": 0, "COST": 0, "TJX": 0, "LOW": 0, 
    "NKE": 0, "SBUX": 0, "K": 0}


    
consumer_goods = black_litterman_optimization(tickers_views_cg, period = "10y")
print(consumer_goods)





# Industrial Materials


# tickers_views_im = {
#     "LIN": 0, "NEM": 0, "SCCO": 0, "RIO": 0, "VMC": 0,
#     "APD": 0, "DD": 0, "VALE": 0, "XOM": 0, "LMT": 0, "MMM": 0,
#     "CAT": 0, "DE": 0, "GE": 0, "CSX": 0, "ECL": 0, "DOW": 0, "FCX": 0, "FLS": 0,
#     "JCI": 0, "AME": 0, "IP": 0, "PKG": 0, "EMR": 0, "X": 0, "GWW": 0, "CNI": 0, 
#     "ITW": 0, "HON": 0, "IEX": 0, "RS": 0, "STLD": 0, "NUE": 0, "VLO": 0,
#     "PHM": 0, "TMO": 0
# }


# industrial_materials = black_litterman_optimization(tickers_views_im, period = "10y")
# print(industrial_materials)







