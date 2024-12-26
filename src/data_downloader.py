import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class DataDownloader:
    def __init__(self):
        self.data_dir = 'data'
        self.ensure_data_directory()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def download_stock_data(self, symbol: str, period: str = '5y') -> pd.DataFrame:
        """Download historical stock data"""
        print(f"Downloading data for {symbol}...")
        stock = yf.Ticker(symbol)
        
        # Get historical market data
        hist_data = stock.history(period=period)
        if hist_data.empty:
            print(f"No historical data found for {symbol}")
            return None
            
        # Get stock info
        info = stock.info
        
        # Add technical indicators
        df = self.add_technical_indicators(hist_data)
        
        # Add fundamental data
        df = self.add_fundamental_data(df, info)
        
        # Add sentiment data from news
        df = self.add_sentiment_data(df, stock)
        
        return df
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        # Trend Indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        
        # Momentum Indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        macd = ta.trend.MACD(df['Close'])
        df['MACD_Line'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Volatility Indicators
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Lower'] = bollinger.bollinger_lband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Volume Indicators
        df['Volume_SMA'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Price Changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Price_Change_20d'] = df['Close'].pct_change(periods=20)
        
        # Volume Changes
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_Change_5d'] = df['Volume'].pct_change(periods=5)
        
        return df
        
    def add_fundamental_data(self, df: pd.DataFrame, info: dict) -> pd.DataFrame:
        """Add fundamental data to the dataframe"""
        # Add fundamental metrics as daily values
        df['PE_Ratio'] = info.get('trailingPE', 0)
        df['EPS'] = info.get('trailingEps', 0)
        df['ROE'] = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
        df['Debt_to_Equity'] = info.get('debtToEquity', 0)
        df['Revenue_Growth'] = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
        df['Dividend_Yield'] = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        df['Market_Cap'] = info.get('marketCap', 0)
        df['Book_Value'] = info.get('bookValue', 0)
        df['Free_Cash_Flow'] = info.get('freeCashflow', 0)
        
        return df
        
    def add_sentiment_data(self, df: pd.DataFrame, stock: yf.Ticker) -> pd.DataFrame:
        """Add sentiment data to the dataframe"""
        try:
            # Get news data
            news = stock.news
            
            # Calculate daily sentiment
            daily_sentiment = {}
            for item in news:
                # Try different timestamp fields
                timestamp = item.get('providerPublishTime') or item.get('publishTime') or item.get('timestamp')
                if timestamp:
                    try:
                        if isinstance(timestamp, (int, float)):
                            date = datetime.fromtimestamp(timestamp).date()
                        else:
                            date = pd.to_datetime(timestamp).date()
                            
                        title = item.get('title', '')
                        summary = item.get('summary', '')
                        sentiment = self.sentiment_analyzer.polarity_scores(title + ' ' + summary)
                        if date not in daily_sentiment:
                            daily_sentiment[date] = []
                        daily_sentiment[date].append(sentiment['compound'])
                    except Exception as e:
                        print(f"Error processing news item: {str(e)}")
                        continue
            
            # Add sentiment to dataframe
            if daily_sentiment:
                df['News_Sentiment'] = df.index.map(lambda x: 
                    np.mean(daily_sentiment.get(x.date(), [0])))
            else:
                print("No sentiment data available, using neutral sentiment")
                df['News_Sentiment'] = 0
            
        except Exception as e:
            print(f"Error adding sentiment data: {str(e)}")
            df['News_Sentiment'] = 0
            
        return df
        
    def download_and_save_data(self, symbols: list):
        """Download and save data for multiple symbols"""
        for symbol in symbols:
            try:
                # Download data
                df = self.download_stock_data(symbol)
                if df is not None:
                    # Convert timezone-aware dates to timezone-naive
                    df.index = df.index.tz_localize(None)
                    
                    # Save to Excel
                    filename = f"{symbol.replace('.', '_')}_data.xlsx"
                    filepath = os.path.join(self.data_dir, filename)
                    df.to_excel(filepath)
                    print(f"Saved data for {symbol} to {filepath}")
                    
                    # Also save a CSV version for easier processing
                    csv_filepath = os.path.join(self.data_dir, f"{symbol.replace('.', '_')}_data.csv")
                    df.to_csv(csv_filepath)
                    print(f"Saved CSV data for {symbol} to {csv_filepath}")
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")

def main():
    # List of Indian stocks to download
    symbols = [
        'RELIANCE.NS',  # Reliance Industries
        'TCS.NS',       # Tata Consultancy Services
        'HDFCBANK.NS',  # HDFC Bank
        'INFY.NS',      # Infosys
        'HINDUNILVR.NS',# Hindustan Unilever
        'ICICIBANK.NS', # ICICI Bank
        'SBIN.NS',      # State Bank of India
        'BHARTIARTL.NS',# Bharti Airtel
        'ITC.NS',       # ITC
        'KOTAKBANK.NS'  # Kotak Mahindra Bank
    ]
    
    # Create downloader and download data
    downloader = DataDownloader()
    downloader.download_and_save_data(symbols)
    print("Data download completed!")

if __name__ == "__main__":
    main() 