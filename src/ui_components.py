import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, Tuple
import yfinance as yf
import pandas as pd

class UIComponents:
    def apply_custom_styling(self):
        """Apply custom styling to the app"""
        st.markdown("""
        <style>
        .glossary-container {
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(40, 40, 80, 0.3);
            border-radius: 8px;
        }
        .glossary-grid {
            display: grid;
            gap: 1rem;
            margin-top: 1rem;
        }
        .glossary-term {
            padding: 0.5rem;
            border-radius: 4px;
            background: rgba(91, 111, 155, 0.5);
        }
        .term {
            font-weight: 600;
            color: white;
            margin-bottom: 0.25rem;
        }
        .definition {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def create_stock_selector(self, stocks: Dict[str, str]) -> str:
        """Create stock selection dropdown"""
        st.markdown("""
        <div class='section-container'>
            <h3>Select Stock</h3>
        </div>
        """, unsafe_allow_html=True)
        
        return st.selectbox(
            "",  # Empty label since we're using custom header
            options=list(stocks.keys()),
            format_func=lambda x: f"{x} ({stocks[x]})"
        )
    
    def display_current_price(self, stock_info: Dict[str, Any]):
        """Display current stock price and change"""
        current_price = stock_info.get('currentPrice', 0)
        previous_close = stock_info.get('previousClose', 0)
        price_change = current_price - previous_close
        price_change_percent = (price_change / previous_close) * 100 if previous_close else 0
        
        change_class = 'positive' if price_change >= 0 else 'negative'
        change_symbol = '▲' if price_change >= 0 else '▼'
        
        st.markdown(f"""
        <div class='price-container'>
            <div class='current-price'>₹{current_price:,.2f}</div>
            <div class='price-change {change_class}'>
                {change_symbol} ₹{abs(price_change):,.2f} ({abs(price_change_percent):.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def display_key_metrics(self, stock_info: Dict[str, Any]):
        """Display key metrics in horizontal boxes"""
        metrics = {
            "Market Cap": stock_info.get("marketCap", "N/A"),
            "P/E Ratio": stock_info.get("trailingPE", "N/A"),
            "EPS": stock_info.get("trailingEps", "N/A"),
            "52W High": stock_info.get("fiftyTwoWeekHigh", "N/A"),
            "52W Low": stock_info.get("fiftyTwoWeekLow", "N/A"),
            "Volume": stock_info.get("volume", "N/A")
        }
        
        # Custom CSS for metric boxes
        st.markdown("""
            <style>
            .metric-container {
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                margin-bottom: 1rem;
            }
            .metric-box {
                background-color: rgba(91, 111, 155, 0.5);
                border-radius: 5px;
                padding: 1rem;
                flex: 1;
                min-width: 150px;
                text-align: center;
            }
            .metric-label {
                color: #CCCCCC;
                font-size: 0.9rem;
                margin-bottom: 0.5rem;
            }
            .metric-value {
                color: white;
                font-size: 1.2rem;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Create metric boxes HTML
        metrics_html = '<div class="metric-container">'
        for label, value in metrics.items():
            if isinstance(value, (int, float)):
                if label == "Market Cap":
                    formatted_value = f"${value:,.0f}"
                elif label == "Volume":
                    formatted_value = f"{value:,.0f}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
                
            metrics_html += f"""
                <div class="metric-box">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
            """
        metrics_html += '</div>'
        
        st.markdown(metrics_html, unsafe_allow_html=True)
        
    def display_technical_indicators(self, stock_info: Dict[str, Any]):
        """Display technical indicators in horizontal boxes"""
        # Get historical data
        close_prices = []
        if 'history' in stock_info:
            hist = stock_info['history']
            if not hist.empty:
                close_prices = hist['Close'].tolist()
        
        # Calculate technical indicators
        indicators = {
            "RSI": self._calculate_rsi(close_prices),
            "MACD": self._calculate_macd(close_prices),
            "Moving Average (50D)": self._calculate_ma(close_prices, 50),
            "Moving Average (200D)": self._calculate_ma(close_prices, 200),
            "Bollinger Bands": self._calculate_bollinger_bands(close_prices)
        }
        
        # Custom CSS for indicator boxes
        st.markdown("""
            <style>
            .indicator-container {
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                margin-bottom: 1rem;
            }
            .indicator-box {
                background-color: rgba(91, 111, 155, 0.5);
                border-radius: 5px;
                padding: 1rem;
                flex: 1;
                min-width: 150px;
                text-align: center;
            }
            .indicator-label {
                color: #CCCCCC;
                font-size: 0.9rem;
                margin-bottom: 0.5rem;
            }
            .indicator-value {
                color: white;
                font-size: 1.2rem;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Create indicator boxes HTML
        indicators_html = '<div class="indicator-container">'
        for label, value in indicators.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.2f}"
            elif isinstance(value, tuple):
                formatted_value = f"Upper: {value[0]:.2f}<br>Lower: {value[1]:.2f}"
            else:
                formatted_value = str(value)
                
            indicators_html += f"""
                <div class="indicator-box">
                    <div class="indicator-label">{label}</div>
                    <div class="indicator-value">{formatted_value}</div>
                </div>
            """
        indicators_html += '</div>'
        
        st.markdown(indicators_html, unsafe_allow_html=True)
    
    def display_glossary(self):
        """Display glossary of terms"""
        st.markdown("""
        <div class='glossary-container'>
            <h3>Glossary</h3>
            <div class='glossary-grid'>
        """, unsafe_allow_html=True)
        
        terms = {
            'Market Capitalization': 'Total value of a company\'s outstanding shares',
            'P/E Ratio': 'Price-to-Earnings ratio, measuring company valuation',
            'EPS': 'Earnings Per Share, measuring company profitability',
            'RSI': 'Relative Strength Index, measuring price momentum',
            'MACD': 'Moving Average Convergence Divergence, trend indicator',
            'Bollinger Bands': 'Volatility indicator showing price channels',
            'Moving Average': 'Average price over a specific time period',
            'Volume': 'Number of shares traded in a given period',
            'Beta': 'Measure of stock\'s volatility compared to market',
            'Volatility': 'Degree of variation in trading price'
        }
        
        for term, definition in terms.items():
            st.markdown(f"""
            <div class='glossary-term'>
                <div class='term'>{term}</div>
                <div class='definition'>{definition}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    def _calculate_rsi(self, data, periods=14):
        """Calculate Relative Strength Index"""
        if not isinstance(data, list) or len(data) < periods:
            return 0
            
        # Convert list to pandas Series if needed
        if isinstance(data, list):
            data = pd.Series(data)
            
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 0
    
    def _calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if not isinstance(data, list) or len(data) < slow:
            return 0
            
        # Convert list to pandas Series if needed
        if isinstance(data, list):
            data = pd.Series(data)
            
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return macd.iloc[-1] - signal_line.iloc[-1]
    
    def _calculate_ma(self, data, period):
        """Calculate Moving Average"""
        if not isinstance(data, list) or len(data) < period:
            return 0
            
        # Convert list to pandas Series if needed
        if isinstance(data, list):
            data = pd.Series(data)
            
        ma = data.rolling(window=period).mean()
        return ma.iloc[-1] if not pd.isna(ma.iloc[-1]) else 0
    
    def _calculate_bollinger_bands(self, data, period=20):
        """Calculate Bollinger Bands"""
        if not isinstance(data, list) or len(data) < period:
            return (0, 0)
            
        # Convert list to pandas Series if needed
        if isinstance(data, list):
            data = pd.Series(data)
            
        ma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = ma + (std * 2)
        lower_band = ma - (std * 2)
        
        return (
            upper_band.iloc[-1] if not pd.isna(upper_band.iloc[-1]) else 0,
            lower_band.iloc[-1] if not pd.isna(lower_band.iloc[-1]) else 0
        ) 