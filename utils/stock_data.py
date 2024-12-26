import yfinance as yf
import pandas as pd

def get_stock_data(symbol):
    """
    Fetch stock data using yfinance
    """
    stock = yf.Ticker(symbol)
    
    # Get historical data
    hist = stock.history(period="2y")
    
    # Get company info
    info = stock.info
    
    return {
        "symbol": symbol,
        "historical_data": hist,
        "info": info,
    } 