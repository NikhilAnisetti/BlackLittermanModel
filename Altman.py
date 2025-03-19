import yfinance as yf
import pandas as pd

def calculate_rsi(ticker, period="6mo", window=14):
    """
    Calculates the Relative Strength Index (RSI) for a given stock ticker.
    
    :param ticker: Stock ticker symbol (e.g., "AAPL")
    :param period: Time period for historical data (default: "6mo")
    :param window: RSI calculation window (default: 14)
    :return: Latest RSI value
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    if hist.empty:
        return None  # No data available

    # Compute price changes
    delta = hist["Close"].diff()

    # Calculate gains and losses
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # Compute RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1] if not rsi.isna().all() else None

# List of stock tickers to check
tickers = ["AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "META", "AMZN", "TSLA", "AMD", "IBM", 
 "ORCL", "CRM", "INTC", "CSCO", "ADBE", "SAP", "TXN", "AVGO", "QCOM", "NOW", 
 "ASML", "ZM", "SHOP", "SNOW", "PLTR", "UBER", "SQ", "NET", "ARKK", "TWLO", 
 "DOCU", "DDOG", "CRWD", "ROKU", "MDB", "PINS", "FSLY", "AI", "BBAI", "SOUN", 
 "ERIC", "NOK", "INFY", "WIT", "GLBE", "ADSK", "TEAM", "WDAY", "RBLX", "ESTC", 
 "U", "AFRM", "OKTA", "ZI", "DUOL", "FVRR", "SE", "BIDU", "JD", "BABA", "BYND", 
 "CHWY", "PRTC.L", "OCDO.L", "AMS", "IFX", "STM", "NXPI", "ROG.SW", "UBI.PA", 
 "AUTO.L", "XPEV", "LI", "NTES"]


# Find overbought stocks
overbought_stocks = []
for ticker in tickers:
    rsi_value = calculate_rsi(ticker)
    if rsi_value is not None and rsi_value > 70:  # Overbought condition
        overbought_stocks.append({"Ticker": ticker, "RSI": rsi_value})

# Convert to DataFrame & Display
df = pd.DataFrame(overbought_stocks)
print(df)
