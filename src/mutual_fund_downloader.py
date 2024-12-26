import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ta

class MutualFundDownloader:
    def __init__(self):
        self.data_dir = 'data/mutual_funds'
        self.ensure_data_directory()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def download_mf_data(self, symbol: str, period: str = '5y') -> pd.DataFrame:
        """Download mutual fund data"""
        print(f"Downloading data for {symbol}...")
        mf = yf.Ticker(symbol)
        
        # Get historical NAV data
        hist_data = mf.history(period=period)
        if hist_data.empty:
            print(f"No historical data found for {symbol}")
            return None
            
        # Get fund info
        info = mf.info
        
        # Add technical indicators
        df = self.add_technical_indicators(hist_data)
        
        # Add fundamental data
        df = self.add_fundamental_data(df, info)
        
        # Add sentiment data from news
        df = self.add_sentiment_data(df, symbol)
        
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
        
        # Returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Monthly_Return'] = df['Close'].pct_change(periods=21)
        df['Yearly_Return'] = df['Close'].pct_change(periods=252)
        
        return df
        
    def add_fundamental_data(self, df: pd.DataFrame, info: dict) -> pd.DataFrame:
        """Add fundamental data to the dataframe"""
        # Add mutual fund specific metrics
        df['AUM'] = info.get('totalAssets', 0)
        df['Expense_Ratio'] = info.get('annualReportExpenseRatio', 0)
        df['Category'] = info.get('category', '')
        df['Beta'] = info.get('beta3Year', 0)
        df['Alpha'] = info.get('alpha3Year', 0)
        df['Sharpe_Ratio'] = info.get('sharpeRatio', 0)
        df['Standard_Deviation'] = info.get('standardDeviation', 0)
        df['YTD_Return'] = info.get('ytdReturn', 0)
        df['Three_Year_Return'] = info.get('threeYearAverageReturn', 0)
        df['Five_Year_Return'] = info.get('fiveYearAverageReturn', 0)
        
        return df
        
    def add_sentiment_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add sentiment data from Google News"""
        try:
            # Fetch news from Google News
            news_items = self.fetch_google_news(symbol)
            
            # Calculate daily sentiment
            daily_sentiment = {}
            for item in news_items:
                date = item['date']
                sentiment = self.sentiment_analyzer.polarity_scores(item['title'] + ' ' + item['snippet'])
                if date not in daily_sentiment:
                    daily_sentiment[date] = []
                daily_sentiment[date].append(sentiment['compound'])
            
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
        
    def fetch_google_news(self, symbol: str, days: int = 30) -> list:
        """Fetch news from Google News"""
        news_items = []
        
        try:
            # Remove .NS suffix for search
            search_term = symbol.replace('.NS', '') + ' mutual fund'
            url = f"https://news.google.com/rss/search?q={search_term}&hl=en-IN&gl=IN&ceid=IN:en"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'xml')
            
            for item in soup.find_all('item'):
                pub_date = pd.to_datetime(item.pubDate.text).date()
                news_items.append({
                    'title': item.title.text,
                    'snippet': item.description.text,
                    'date': pub_date,
                    'link': item.link.text
                })
                
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            
        return news_items
        
    def download_and_save_data(self, symbols: list):
        """Download and save data for multiple mutual funds"""
        for symbol in symbols:
            try:
                # Download data
                df = self.download_mf_data(symbol)
                if df is not None:
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
    # List of Indian Mutual Funds
    symbols = [
        'HDFC500.BO',    # HDFC Top 100 Fund
        'ICICI500.BO',   # ICICI Prudential Bluechip Fund
        'AXIS500.BO',    # Axis Bluechip Fund
        'SBI500.BO',     # SBI Bluechip Fund
        'TATA500.BO',    # Tata Large Cap Fund
        'KOTAKSTD.BO',   # Kotak Standard Multicap Fund
        'ABSL500.BO',    # Aditya Birla Sun Life Frontline Equity Fund
        'UTI500.BO',     # UTI Equity Fund
        'ICICINF50.BO',  # ICICI Prudential Nifty 50 Index Fund
        'HDFCNF50.BO'    # HDFC Index Fund-NIFTY 50 Plan
    ]
    
    # Create downloader and download data
    downloader = MutualFundDownloader()
    downloader.download_and_save_data(symbols)
    print("Data download completed!")

if __name__ == "__main__":
    main() 