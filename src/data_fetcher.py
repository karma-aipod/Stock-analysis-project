import yfinance as yf
import pandas as pd
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime, timedelta
from gnews import GNews
import time

class StockDataFetcher:
    def __init__(self):
        self.news_client = GNews(language='en', country='IN', period='7d', max_results=10)
        pass
    
    def get_stock_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Fetch historical stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed stock information"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Safe get function to handle callable attributes
            def safe_get(d: dict, key: str, default: Any = None) -> Any:
                value = d.get(key, default)
                if callable(value):
                    try:
                        return value()
                    except:
                        return default
                return value
            
            # Create a new dict with safely accessed values
            safe_info = {}
            for key, value in info.items():
                safe_info[key] = safe_get(info, key)
            
            # Calculate additional metrics
            hist = stock.history(period='1y')
            if not hist.empty:
                # Calculate volatility
                daily_returns = hist['Close'].pct_change()
                safe_info['dailyVolatility'] = daily_returns.std() * 100
                safe_info['annualVolatility'] = safe_info['dailyVolatility'] * np.sqrt(252)
                
                # Calculate traded value
                safe_info['tradedValue'] = hist['Close'].iloc[-1] * hist['Volume'].iloc[-1]
                
                # Calculate 52-week metrics
                safe_info['fiftyTwoWeekHigh'] = hist['High'].max()
                safe_info['fiftyTwoWeekLow'] = hist['Low'].min()
            
            return safe_info
        except Exception as e:
            print(f"Error fetching stock info: {str(e)}")
            return {}
    
    def get_sector_performance(self) -> Dict[str, float]:
        """Get sector performance data"""
        sectors = {
            'XLF': 'Financial',
            'XLK': 'Technology',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrial',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
        
        performance = {}
        for symbol, sector in sectors.items():
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='1mo')
                if not hist.empty:
                    start_price = hist['Close'].iloc[0]
                    end_price = hist['Close'].iloc[-1]
                    perf = ((end_price - start_price) / start_price) * 100
                    performance[sector] = perf
                else:
                    performance[sector] = 0
            except:
                performance[sector] = 0
        
        return performance
    
    def get_market_share_data(self, symbol: str) -> Dict[str, float]:
        """Get market share data for the company and its competitors"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            sector = info.get('sector', '')
            
            if not sector:
                return {}
            
            # Get peer companies
            peers = self._get_peer_companies(sector)
            total_market_cap = 0
            market_shares = {}
            
            # Calculate total market cap
            for peer in peers:
                try:
                    peer_stock = yf.Ticker(peer)
                    peer_info = peer_stock.info
                    market_cap = peer_info.get('marketCap', 0)
                    if market_cap > 0:
                        total_market_cap += market_cap
                        market_shares[peer_info.get('shortName', peer)] = market_cap
                except:
                    continue
            
            # Convert to percentages
            if total_market_cap > 0:
                market_shares = {k: (v/total_market_cap)*100 for k, v in market_shares.items()}
            
            return market_shares
            
        except Exception as e:
            print(f"Error getting market share data: {str(e)}")
            return {}
    
    def get_competitor_data(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """Get competitor performance data"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            sector = info.get('sector', '')
            
            if not sector:
                return {}
            
            # Get peer companies
            peers = self._get_peer_companies(sector)
            competitor_data = {}
            
            for peer in peers:
                try:
                    peer_stock = yf.Ticker(peer)
                    peer_info = peer_stock.info
                    
                    competitor_data[peer_info.get('shortName', peer)] = {
                        'Revenue Growth': peer_info.get('revenueGrowth', 0) * 100 if peer_info.get('revenueGrowth') else 0,
                        'Profit Margin': peer_info.get('profitMargins', 0) * 100 if peer_info.get('profitMargins') else 0,
                        'Market Share': 0  # Will be updated with actual market share
                    }
                except:
                    continue
            
            # Add market share data
            market_shares = self.get_market_share_data(symbol)
            for company in competitor_data:
                competitor_data[company]['Market Share'] = market_shares.get(company, 0)
            
            return competitor_data
            
        except Exception as e:
            print(f"Error getting competitor data: {str(e)}")
            return {}
    
    def get_stock_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Get news related to the stock"""
        try:
            # Get company name from symbol
            stock = yf.Ticker(symbol)
            company_name = stock.info.get('shortName', '').split()[0]  # Get first word of company name
            
            # Get news from both yfinance and Google News
            yf_news = self._get_yf_news(stock)
            google_news = self._get_google_news(company_name)
            
            # Combine and sort news by date
            all_news = yf_news + google_news
            all_news.sort(key=lambda x: x['publish_date'], reverse=True)
            
            return all_news
            
        except Exception as e:
            print(f"Error getting news: {str(e)}")
            return []
    
    def _get_yf_news(self, stock: yf.Ticker) -> List[Dict[str, Any]]:
        """Get news from yfinance"""
        try:
            news = stock.news
            processed_news = []
            
            for item in news:
                processed_news.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'publish_date': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'type': 'Financial News',
                    'thumbnail': item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', ''),
                    'summary': item.get('summary', '')
                })
            
            return processed_news
        except:
            return []
    
    def _get_google_news(self, company_name: str) -> List[Dict[str, Any]]:
        """Get news from Google News"""
        try:
            # Search for company news
            news_items = self.news_client.get_news(f"{company_name} stock market")
            processed_news = []
            
            for item in news_items:
                # Add a small delay to avoid rate limiting
                time.sleep(0.1)
                
                # Get article
                article = self.news_client.get_full_article(item['url'])
                
                processed_news.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', {}).get('title', 'Google News'),
                    'link': item.get('url', ''),
                    'publish_date': datetime.strptime(item.get('published date', ''), '%a, %d %b %Y %H:%M:%S GMT'),
                    'type': 'General News',
                    'thumbnail': '',  # Google News doesn't provide thumbnails
                    'summary': article.text[:500] + '...' if article and article.text else item.get('description', '')
                })
            
            return processed_news
        except Exception as e:
            print(f"Error fetching Google News: {str(e)}")
            return []
    
    def _get_peer_companies(self, sector: str, limit: int = 5) -> List[str]:
        """Get list of peer companies in the same sector"""
        try:
            # For demonstration, using a simple list of major companies per sector
            sector_companies = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
                'Financial': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
                'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO'],
                'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
                'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
                'Industrial': ['HON', 'UPS', 'BA', 'CAT', 'GE'],
                'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
                'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA']
            }
            
            return sector_companies.get(sector, [])[:limit]
            
        except Exception as e:
            print(f"Error getting peer companies: {str(e)}")
            return [] 