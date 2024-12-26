def get_nifty50_stocks():
    """Return a dictionary of NIFTY 50 stocks"""
    return {
        "Reliance Industries": "RELIANCE.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Infosys": "INFY.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "TCS": "TCS.NS",
        "Hindustan Unilever": "HINDUNILVR.NS",
        "ITC": "ITC.NS",
        "State Bank of India": "SBIN.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS",
        "Axis Bank": "AXISBANK.NS",
        "Larsen & Toubro": "LT.NS",
        "Asian Paints": "ASIANPAINT.NS",
        "HCL Technologies": "HCLTECH.NS",
        "Maruti Suzuki": "MARUTI.NS",
        "Sun Pharma": "SUNPHARMA.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
        "Wipro": "WIPRO.NS",
        "UltraTech Cement": "ULTRACEMCO.NS",
        "Power Grid": "POWERGRID.NS"
    }

def get_global_stocks():
    """Return a dictionary of global stocks"""
    return {
        # US Tech
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "Meta": "META",
        
        # US Finance
        "JPMorgan Chase": "JPM",
        "Bank of America": "BAC",
        "Goldman Sachs": "GS",
        
        # European Stocks
        "ASML": "ASML",
        "SAP": "SAP",
        "LVMH": "MC.PA",
        "Nestle": "NESN.SW",
        
        # Asian Stocks
        "Samsung": "005930.KS",
        "Toyota": "7203.T",
        "Alibaba": "9988.HK",
        "Taiwan Semi": "TSM"
    }

def get_default_stocks():
    """Return a dictionary of default stocks when API fails"""
    return {
        "Tesla": "TSLA",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Reliance Industries": "RELIANCE.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Infosys": "INFY.NS"
    } 