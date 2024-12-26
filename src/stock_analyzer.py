import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import ta
from plotly.subplots import make_subplots

class StockAnalyzer:
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def calculate_fundamental_metrics(self, stock_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate fundamental metrics for stock analysis"""
        safe_get = lambda d, k, default: d.get(k, default) if not callable(d.get(k)) else default
        
        metrics = {
            # Valuation Metrics
            'PE_Ratio': safe_get(stock_info, 'trailingPE', 0),
            'Forward_PE': safe_get(stock_info, 'forwardPE', 0),
            'PEG_Ratio': safe_get(stock_info, 'pegRatio', 0),
            'Price_to_Book': safe_get(stock_info, 'priceToBook', 0),
            'Price_to_Sales': safe_get(stock_info, 'priceToSalesTrailing12Months', 0),
            'EV_to_EBITDA': safe_get(stock_info, 'enterpriseToEbitda', 0),
            
            # Profitability Metrics
            'EPS': safe_get(stock_info, 'trailingEps', 0),
            'Forward_EPS': safe_get(stock_info, 'forwardEps', 0),
            'EPS_Growth': safe_get(stock_info, 'earningsQuarterlyGrowth', 0) * 100,
            'ROE': (safe_get(stock_info, 'netIncome', 0) / safe_get(stock_info, 'totalStockholderEquity', 1)) * 100 if safe_get(stock_info, 'totalStockholderEquity', 0) != 0 else 0,
            'ROA': safe_get(stock_info, 'returnOnAssets', 0) * 100,
            'ROCE': safe_get(stock_info, 'returnOnCapital', 0) * 100,
            'Operating_Margin': safe_get(stock_info, 'operatingMargins', 0) * 100,
            'Net_Profit_Margin': safe_get(stock_info, 'profitMargins', 0) * 100,
            
            # Financial Health Metrics
            'Current_Ratio': safe_get(stock_info, 'currentRatio', 0),
            'Quick_Ratio': safe_get(stock_info, 'quickRatio', 0),
            'Debt_to_Equity': safe_get(stock_info, 'debtToEquity', 0),
            'Interest_Coverage': safe_get(stock_info, 'interestCoverage', 0),
            'Free_Cash_Flow': safe_get(stock_info, 'freeCashflow', 0) / 10000000,  # Convert to Cr
            'Operating_Cash_Flow': safe_get(stock_info, 'operatingCashflow', 0) / 10000000,  # Convert to Cr
            
            # Growth Metrics
            'Revenue_Growth': safe_get(stock_info, 'revenueGrowth', 0) * 100,
            'Earnings_Growth': safe_get(stock_info, 'earningsGrowth', 0) * 100,
            'Free_Cash_Flow_Growth': safe_get(stock_info, 'freeCashflowGrowth', 0) * 100,
            
            # Dividend Metrics
            'Dividend_Yield': safe_get(stock_info, 'dividendYield', 0) * 100,
            'Payout_Ratio': safe_get(stock_info, 'payoutRatio', 0) * 100,
            'Dividend_Growth': safe_get(stock_info, 'dividendGrowth', 0) * 100,
            
            # Market Metrics
            'Market_Cap': safe_get(stock_info, 'marketCap', 0) / 10000000,  # Convert to Cr
            'Enterprise_Value': safe_get(stock_info, 'enterpriseValue', 0) / 10000000,  # Convert to Cr
            'Beta': safe_get(stock_info, 'beta', 0),
            'Shares_Outstanding': safe_get(stock_info, 'sharesOutstanding', 0),
            'Float_Shares': safe_get(stock_info, 'floatShares', 0),
            'Institutional_Ownership': safe_get(stock_info, 'institutionPercentHeld', 0) * 100,
            
            # Per Share Metrics
            'Book_Value_Per_Share': safe_get(stock_info, 'bookValue', 0),
            'Cash_Per_Share': safe_get(stock_info, 'totalCash', 0) / safe_get(stock_info, 'sharesOutstanding', 1) if safe_get(stock_info, 'sharesOutstanding', 0) != 0 else 0,
            'Revenue_Per_Share': safe_get(stock_info, 'revenuePerShare', 0)
        }
        
        return metrics
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for analysis"""
        try:
            if data.empty or len(data) < 20:  # Need at least 20 days of data
                print("Insufficient data for technical analysis")
                return pd.DataFrame()
            
            # Verify required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                print(f"Missing required columns. Available columns: {data.columns.tolist()}")
                return pd.DataFrame()
            
            # Check for valid price data
            if (data['Close'] <= 0).any():
                print("Invalid price data detected (zero or negative values)")
                data = data[data['Close'] > 0]  # Filter out invalid prices
            
            df = data.copy()
            
            # Calculate Daily Returns
            try:
                df['Daily_Return'] = df['Close'].pct_change()
                # Handle any NaN values in the first row
                df['Daily_Return'] = df['Daily_Return'].fillna(0)
                
                # Add cumulative returns
                df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
            except Exception as e:
                print(f"Error calculating Daily Returns: {str(e)}")
                # Fallback calculation
                df['Daily_Return'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
                df['Daily_Return'] = df['Daily_Return'].fillna(0)
                df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
            
            try:
                # Moving Averages
                df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
                df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
                df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
                
                # MACD - with error handling
                try:
                    # Calculate EMAs
                    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
                    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
                    
                    # Calculate MACD line
                    df['MACD_Line'] = ema12 - ema26
                    
                    # Calculate Signal line (9-day EMA of MACD)
                    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
                    
                    # Calculate MACD Histogram
                    df['MACD_Histogram'] = df['MACD_Line'] - df['MACD_Signal']
                    
                    # Fill any NaN values
                    df['MACD_Line'] = df['MACD_Line'].ffill().bfill()
                    df['MACD_Signal'] = df['MACD_Signal'].ffill().bfill()
                    df['MACD_Histogram'] = df['MACD_Histogram'].ffill().bfill()
                    
                except Exception as e:
                    print(f"Error calculating MACD: {str(e)}")
                    # Set default values if calculation fails
                    df['MACD_Line'] = 0
                    df['MACD_Signal'] = 0
                    df['MACD_Histogram'] = 0
                
                # RSI
                df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
                
                # Price Rate of Change (ROC)
                try:
                    df['Price_ROC'] = ta.momentum.ROCIndicator(df['Close'], window=12).roc()
                except Exception as e:
                    print(f"Error calculating Price ROC: {str(e)}")
                    # Fallback calculation for ROC
                    df['Price_ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
                
                # Stochastic Oscillator
                try:
                    stoch = ta.momentum.StochasticOscillator(
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        window=14,
                        smooth_window=3
                    )
                    df['Stochastic_K'] = stoch.stoch()
                    df['Stochastic_D'] = stoch.stoch_signal()
                except Exception as e:
                    print(f"Error calculating Stochastic: {str(e)}")
                    # Fallback calculation
                    window = 14
                    low_min = df['Low'].rolling(window=window).min()
                    high_max = df['High'].rolling(window=window).max()
                    df['Stochastic_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
                    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
                
                # Bollinger Bands
                bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
                df['BB_Upper'] = bollinger.bollinger_hband()
                df['BB_Lower'] = bollinger.bollinger_lband()
                df['BB_Middle'] = bollinger.bollinger_mavg()
                
                # Volume Indicators
                df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
                
                # On Balance Volume (OBV)
                try:
                    # Check for required data
                    if 'Close' not in df.columns or 'Volume' not in df.columns:
                        raise ValueError("Missing required columns for OBV calculation")
                    
                    # Handle any zero or negative values
                    if (df['Volume'] <= 0).any():
                        df.loc[df['Volume'] <= 0, 'Volume'] = df['Volume'].mean()
                    
                    # Primary OBV calculation
                    try:
                        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
                            close=df['Close'],
                            volume=df['Volume']
                        ).on_balance_volume()
                    except Exception as e:
                        print(f"Primary OBV calculation failed: {str(e)}")
                        raise  # Try fallback calculation
                    
                    # Verify OBV calculation
                    if df['OBV'].isnull().any():
                        raise ValueError("OBV calculation produced null values")
                        
                except Exception as e:
                    print(f"Using fallback OBV calculation: {str(e)}")
                    # Fallback calculation for OBV
                    obv = 0
                    obv_list = []
                    
                    for i in range(len(df)):
                        if i == 0:
                            obv_list.append(obv)
                            continue
                            
                        current_close = df['Close'].iloc[i]
                        prev_close = df['Close'].iloc[i-1]
                        current_volume = df['Volume'].iloc[i]
                        
                        if current_close > prev_close:
                            obv += current_volume
                        elif current_close < prev_close:
                            obv -= current_volume
                        # If prices are equal, OBV remains the same
                        
                        obv_list.append(obv)
                    
                    df['OBV'] = obv_list
                    
                    # Verify fallback calculation
                    if df['OBV'].isnull().any():
                        print("Warning: Fallback OBV calculation produced null values")
                        df['OBV'] = df['OBV'].fillna(method='ffill').fillna(method='bfill')
                
                # ADX
                try:
                    adx = ta.trend.ADXIndicator(
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        window=14
                    )
                    df['ADX'] = adx.adx()
                    df['DI_Plus'] = adx.adx_pos()
                    df['DI_Minus'] = adx.adx_neg()
                except Exception as e:
                    print(f"Error calculating ADX: {str(e)}")
                    # Fallback calculation for ADX
                    tr1 = df['High'] - df['Low']
                    tr2 = abs(df['High'] - df['Close'].shift(1))
                    tr3 = abs(df['Low'] - df['Close'].shift(1))
                    tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
                    atr = tr.rolling(window=14).mean()
                    df['ADX'] = atr / df['Close'] * 100  # Simplified ADX calculation
                    df['DI_Plus'] = df['ADX']  # Placeholder
                    df['DI_Minus'] = df['ADX']  # Placeholder
                
                # ATR
                df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
                
                # Ichimoku Cloud
                try:
                    # Calculate Tenkan-sen (Conversion Line)
                    high_9 = df['High'].rolling(window=9).max()
                    low_9 = df['Low'].rolling(window=9).min()
                    df['Ichimoku_Conversion_Line'] = (high_9 + low_9) / 2
                    
                    # Calculate Kijun-sen (Base Line)
                    high_26 = df['High'].rolling(window=26).max()
                    low_26 = df['Low'].rolling(window=26).min()
                    df['Ichimoku_Base_Line'] = (high_26 + low_26) / 2
                    
                    # Calculate Senkou Span A (Leading Span A)
                    df['Ichimoku_Leading_Span_A'] = ((df['Ichimoku_Conversion_Line'] + df['Ichimoku_Base_Line']) / 2).shift(26)
                    
                    # Calculate Senkou Span B (Leading Span B)
                    high_52 = df['High'].rolling(window=52).max()
                    low_52 = df['Low'].rolling(window=52).min()
                    df['Ichimoku_Leading_Span_B'] = ((high_52 + low_52) / 2).shift(26)
                    
                    # Calculate Chikou Span (Lagging Span)
                    df['Ichimoku_Lagging_Span'] = df['Close'].shift(-26)
                    
                except Exception as e:
                    print(f"Error calculating Ichimoku Cloud: {str(e)}")
                    # Set default values
                    df['Ichimoku_Conversion_Line'] = df['Close'].rolling(window=9).mean()
                    df['Ichimoku_Base_Line'] = df['Close'].rolling(window=26).mean()
                    df['Ichimoku_Leading_Span_A'] = df['Close'].rolling(window=26).mean()
                    df['Ichimoku_Leading_Span_B'] = df['Close'].rolling(window=52).mean()
                    df['Ichimoku_Lagging_Span'] = df['Close'].shift(-26)
                
                # Chaikin Money Flow (CMF)
                try:
                    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        volume=df['Volume'],
                        window=20
                    ).chaikin_money_flow()
                except Exception as e:
                    print(f"Error calculating CMF: {str(e)}")
                    # Fallback calculation for CMF
                    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
                    mf_volume = mf_multiplier * df['Volume']
                    df['CMF'] = mf_volume.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
                
                # Volatility
                df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
                
                # Fill NaN values
                df = df.bfill().ffill()
                
                # Verify data quality
                if df['Close'].iloc[-1] <= 0:
                    print("Warning: Current price is zero or negative")
                    return pd.DataFrame()
                
                # Print diagnostics
                print(f"Latest price: {df['Close'].iloc[-1]:.2f}")
                print(f"MACD Line: {df['MACD_Line'].iloc[-1]:.2f}")
                print(f"MACD Signal: {df['MACD_Signal'].iloc[-1]:.2f}")
                print(f"ADX: {df['ADX'].iloc[-1]:.2f}")
                print(f"Stochastic K: {df['Stochastic_K'].iloc[-1]:.2f}")
                print(f"Stochastic D: {df['Stochastic_D'].iloc[-1]:.2f}")
                
                return df
                
            except Exception as e:
                print(f"Error calculating technical indicators: {str(e)}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Critical error in calculate_technical_indicators: {str(e)}")
            return pd.DataFrame()
    
    def analyze_stock_health(self, stock_info: Dict[str, Any], technical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall stock health using both fundamental and technical indicators"""
        # Get fundamental metrics
        fundamental_metrics = self.calculate_fundamental_metrics(stock_info)
        
        # Get latest technical indicators
        latest_data = technical_data.iloc[-1]
        
        # Analyze price trend
        price_trend = self._analyze_price_trend(technical_data)
        
        # Analyze momentum
        momentum_analysis = self._analyze_momentum(latest_data)
        
        # Analyze volatility
        volatility_analysis = self._analyze_volatility(latest_data, technical_data)
        
        # Volume analysis
        volume_analysis = self._analyze_volume(latest_data, technical_data)
        
        # Technical signal analysis
        technical_signals = self._analyze_technical_signals(latest_data)
        
        # Combine all analyses
        analysis = {
            'Fundamental_Metrics': fundamental_metrics,
            'Technical_Analysis': {
                'Price_Trend': price_trend['trend'],
                'Trend_Strength': price_trend['strength'],
                'Support_Level': price_trend['support'],
                'Resistance_Level': price_trend['resistance'],
                'Momentum_Status': momentum_analysis['status'],
                'Momentum_Strength': momentum_analysis['strength'],
                'Volatility_Status': volatility_analysis['status'],
                'Volatility_Change': volatility_analysis['change'],
                'Volume_Trend': volume_analysis['trend'],
                'Volume_Strength': volume_analysis['strength'],
                'Money_Flow': volume_analysis['money_flow'],
                'Technical_Signals': technical_signals
            },
            'Risk_Metrics': {
                'Beta': fundamental_metrics['Beta'],
                'Volatility': latest_data['Volatility'],
                'ATR': latest_data['ATR'],
                'Downside_Risk': self._calculate_downside_risk(technical_data),
                'VaR_95': self._calculate_var(technical_data, 0.95),
                'Sharpe_Ratio': self._calculate_sharpe_ratio(technical_data)
            },
            'Market_Sentiment': {
                'RSI_Signal': 'Overbought' if latest_data['RSI'] > 70 else 'Oversold' if latest_data['RSI'] < 30 else 'Neutral',
                'MACD_Signal': 'Buy' if latest_data['MACD_Line'] > latest_data['MACD_Signal'] else 'Sell',
                'ADX_Trend_Strength': 'Strong' if latest_data['ADX'] > 25 else 'Weak',
                'Ichimoku_Signal': self._analyze_ichimoku(latest_data),
                'CMF_Signal': 'Positive' if latest_data['CMF'] > 0 else 'Negative'
            },
            'Overall_Score': self._calculate_overall_score(fundamental_metrics, latest_data, technical_data)
        }
        
        return analysis
        
    def _analyze_price_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trend using multiple indicators"""
        latest = data.iloc[-1]
        sma_20 = latest['SMA_20']
        sma_50 = latest['SMA_50']
        sma_200 = latest['SMA_200']
        current_price = latest['Close']
        
        # Determine trend
        if current_price > sma_20 > sma_50 > sma_200:
            trend = "Strong Uptrend"
        elif current_price > sma_20 and current_price > sma_50:
            trend = "Uptrend"
        elif current_price < sma_20 < sma_50 < sma_200:
            trend = "Strong Downtrend"
        elif current_price < sma_20 and current_price < sma_50:
            trend = "Downtrend"
        else:
            trend = "Sideways"
        
        # Calculate trend strength using ADX
        strength = "Strong" if latest['ADX'] > 25 else "Moderate" if latest['ADX'] > 20 else "Weak"
        
        # Find support and resistance using Fibonacci levels with error handling
        try:
            fib_levels = [
                latest['Fib_61.8'],
                latest['Fib_50.0'],
                latest['Fib_38.2'],
                latest['Fib_23.6']
            ]
            
            support_levels = [level for level in fib_levels if level < current_price]
            resistance_levels = [level for level in fib_levels if level > current_price]
            
            support = max(support_levels) if support_levels else current_price * 0.95  # Default to 5% below current price
            resistance = min(resistance_levels) if resistance_levels else current_price * 1.05  # Default to 5% above current price
            
        except Exception:
            # Fallback values if Fibonacci calculations fail
            support = current_price * 0.95  # Default to 5% below current price
            resistance = current_price * 1.05  # Default to 5% above current price
        
        return {
            'trend': trend,
            'strength': strength,
            'support': support,
            'resistance': resistance
        }
        
    def _analyze_momentum(self, latest_data: pd.Series) -> Dict[str, str]:
        """Analyze momentum using multiple indicators"""
        rsi = latest_data['RSI']
        stoch_k = latest_data['Stochastic_K']
        stoch_d = latest_data['Stochastic_D']
        roc = latest_data['Price_ROC']
        
        # Determine momentum status
        if rsi > 70 and stoch_k > 80:
            status = "Strongly Overbought"
        elif rsi < 30 and stoch_k < 20:
            status = "Strongly Oversold"
        elif rsi > 60:
            status = "Overbought"
        elif rsi < 40:
            status = "Oversold"
        else:
            status = "Neutral"
        
        # Determine momentum strength
        if abs(roc) > 5 and abs(stoch_k - stoch_d) > 10:
            strength = "Strong"
        elif abs(roc) > 2:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        return {
            'status': status,
            'strength': strength
        }
        
    def _analyze_volatility(self, latest_data: pd.Series, historical_data: pd.DataFrame) -> Dict[str, str]:
        """Analyze volatility using multiple indicators"""
        current_volatility = latest_data['Volatility']
        avg_volatility = historical_data['Volatility'].mean()
        volatility_change = (current_volatility - avg_volatility) / avg_volatility * 100
        
        if current_volatility > 30:
            status = "High"
        elif current_volatility > 15:
            status = "Moderate"
        else:
            status = "Low"
            
        return {
            'status': status,
            'change': f"{volatility_change:+.1f}%"
        }
        
    def _analyze_volume(self, latest_data: pd.Series, historical_data: pd.DataFrame) -> Dict[str, str]:
        """Analyze volume and money flow"""
        volume_ratio = latest_data['Volume_Ratio']
        cmf = latest_data['CMF']
        obv_change = (latest_data['OBV'] - historical_data['OBV'].shift(1).iloc[-1]) / abs(historical_data['OBV'].shift(1).iloc[-1]) * 100
        
        # Determine volume trend
        if volume_ratio > 2:
            trend = "Strongly Above Average"
        elif volume_ratio > 1.5:
            trend = "Above Average"
        elif volume_ratio < 0.5:
            trend = "Below Average"
        else:
            trend = "Average"
        
        # Determine volume strength
        if abs(obv_change) > 5:
            strength = "Strong"
        elif abs(obv_change) > 2:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        # Determine money flow
        money_flow = "Strong Buying" if cmf > 0.2 else "Buying" if cmf > 0 else "Strong Selling" if cmf < -0.2 else "Selling"
        
        return {
            'trend': trend,
            'strength': strength,
            'money_flow': money_flow
        }
        
    def _analyze_technical_signals(self, latest_data: pd.Series) -> Dict[str, str]:
        """Analyze various technical signals"""
        signals = {}
        
        # MACD Signal
        signals['MACD'] = "Buy" if latest_data['MACD_Line'] > latest_data['MACD_Signal'] else "Sell"
        
        # RSI Signal
        if latest_data['RSI'] > 70:
            signals['RSI'] = "Sell"
        elif latest_data['RSI'] < 30:
            signals['RSI'] = "Buy"
        else:
            signals['RSI'] = "Neutral"
        
        # Bollinger Bands Signal
        price = latest_data['Close']
        if price > latest_data['BB_Upper']:
            signals['Bollinger'] = "Sell"
        elif price < latest_data['BB_Lower']:
            signals['Bollinger'] = "Buy"
        else:
            signals['Bollinger'] = "Neutral"
        
        # ADX Signal
        if latest_data['ADX'] > 25:
            if latest_data['DI_Plus'] > latest_data['DI_Minus']:
                signals['ADX'] = "Strong Buy"
            else:
                signals['ADX'] = "Strong Sell"
        else:
            signals['ADX'] = "Neutral"
        
        return signals
        
    def _analyze_ichimoku(self, latest_data: pd.Series) -> str:
        """Analyze Ichimoku Cloud signals"""
        price = latest_data['Close']
        conversion = latest_data['Ichimoku_Conversion_Line']
        base = latest_data['Ichimoku_Base_Line']
        span_a = latest_data['Ichimoku_Leading_Span_A']
        span_b = latest_data['Ichimoku_Leading_Span_B']
        
        if price > span_a and price > span_b:
            if conversion > base:
                return "Strong Buy"
            else:
                return "Buy"
        elif price < span_a and price < span_b:
            if conversion < base:
                return "Strong Sell"
            else:
                return "Sell"
        else:
            return "Neutral"
        
    def _calculate_downside_risk(self, data: pd.DataFrame) -> float:
        """Calculate downside risk using negative returns only"""
        returns = data['Daily_Return'].dropna()
        negative_returns = returns[returns < 0]
        return negative_returns.std() * np.sqrt(252) * 100
        
    def _calculate_var(self, data: pd.DataFrame, confidence: float) -> float:
        """Calculate Value at Risk"""
        returns = data['Daily_Return'].dropna()
        return abs(np.percentile(returns, (1 - confidence) * 100)) * 100
        
    def _calculate_sharpe_ratio(self, data: pd.DataFrame, risk_free_rate: float = 0.03) -> float:
        """Calculate Sharpe Ratio"""
        returns = data['Daily_Return'].dropna()
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
        
    def _calculate_overall_score(self, fundamental_metrics: Dict[str, float], technical_data: pd.Series, historical_data: pd.DataFrame) -> float:
        """Calculate overall stock score based on fundamental and technical metrics"""
        score = 0
        
        # Fundamental factors (40% weight)
        fundamental_score = 0
        if fundamental_metrics['PE_Ratio'] > 0:
            fundamental_score += (20 / fundamental_metrics['PE_Ratio']) * 10  # Lower P/E is better
        fundamental_score += min(fundamental_metrics['ROE'], 30) / 3  # ROE up to 30% is good
        fundamental_score += min(fundamental_metrics['Revenue_Growth'], 50) / 5  # Growth up to 50% is good
        fundamental_score += min(fundamental_metrics['Operating_Margin'], 30) / 3  # Operating margin up to 30% is good
        fundamental_score += (1 - min(fundamental_metrics['Debt_to_Equity'], 2) / 2) * 10  # Lower D/E is better
        
        # Technical factors (40% weight)
        technical_score = 0
        if 30 <= technical_data['RSI'] <= 70:
            technical_score += 10  # Balanced RSI
        technical_score += 10 if technical_data['MACD_Line'] > technical_data['MACD_Signal'] else 0  # MACD signal
        technical_score += 10 if technical_data['ADX'] > 25 else 5 if technical_data['ADX'] > 20 else 0  # Strong trend
        technical_score += 10 if technical_data['CMF'] > 0 else 0  # Positive money flow
        
        # Risk factors (20% weight)
        risk_score = 0
        volatility = technical_data['Volatility']
        if volatility < 20:
            risk_score += 10
        elif volatility < 30:
            risk_score += 5
        
        beta = fundamental_metrics['Beta']
        if 0.5 <= beta <= 1.5:
            risk_score += 10  # Moderate beta
        
        # Combine scores with weights
        score = (fundamental_score * 0.4) + (technical_score * 0.4) + (risk_score * 0.2)
        
        # Normalize score to 0-100 range
        return min(max(score, 0), 100)

    def plot_sector_performance(self, sector_data: Dict[str, float]) -> go.Figure:
        """Create a sector performance visualization"""
        if not sector_data:
            # Create empty chart with message if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No sector data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig
            
        # Sort sectors by performance
        sectors = list(sector_data.keys())
        performances = list(sector_data.values())
        
        # Create color scale based on performance
        colors = ['#EF553B' if p < 0 else '#00CC96' for p in performances]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=sectors,
                y=performances,
                marker_color=colors,
                text=[f"{p:+.2f}%" for p in performances],
                textposition='auto',
            )
        ])
        
        # Update layout
        fig.update_layout(
            title="Sector Performance",
            xaxis_title="Sectors",
            yaxis_title="Performance (%)",
            template="plotly_dark",
            showlegend=False,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.2)'
            )
        )
        
        return fig

    def plot_market_share(self, market_share_data: Dict[str, float]) -> go.Figure:
        """Create a market share visualization"""
        if not market_share_data:
            # Create empty chart with message if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No market share data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig
            
        # Sort companies by market share
        companies = list(market_share_data.keys())
        shares = list(market_share_data.values())
        
        # Create pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=companies,
                values=shares,
                hole=0.4,
                textinfo='label+percent',
                marker=dict(
                    colors=['#00CC96', '#636EFA', '#EF553B', '#AB63FA', '#FFA15A']
                )
            )
        ])
        
        # Update layout
        fig.update_layout(
            title="Market Share Distribution",
            template="plotly_dark",
            showlegend=True,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

    def generate_portfolio(self, investment_amount: float, risk_tolerance: str, investment_horizon: str) -> Dict[str, Any]:
        """Generate portfolio recommendation"""
        # Define allocation based on risk tolerance
        allocations = {
            "Low": {
                "Large Cap": 0.50,
                "Mid Cap": 0.20,
                "Debt": 0.20,
                "Gold": 0.10
            },
            "Medium": {
                "Large Cap": 0.40,
                "Mid Cap": 0.30,
                "Small Cap": 0.15,
                "International": 0.15
            },
            "High": {
                "Large Cap": 0.30,
                "Mid Cap": 0.30,
                "Small Cap": 0.25,
                "International": 0.15
            }
        }
        
        # Get allocation based on risk tolerance
        allocation = allocations.get(risk_tolerance, allocations["Medium"])
        
        # Calculate expected returns based on historical data
        expected_returns = {
            "Large Cap": 0.12,  # 12% annual return
            "Mid Cap": 0.15,    # 15% annual return
            "Small Cap": 0.18,  # 18% annual return
            "Debt": 0.07,       # 7% annual return
            "Gold": 0.08,       # 8% annual return
            "International": 0.14  # 14% annual return
        }
        
        # Convert investment horizon to years
        years = {
            "1 Year": 1,
            "3 Years": 3,
            "5 Years": 5,
            "10 Years": 10
        }.get(investment_horizon, 5)
        
        # Calculate metrics
        total_return = sum(allocation.get(k, 0) * expected_returns.get(k, 0) for k in allocation)
        projected_value = investment_amount * 12 * years * (1 + total_return) ** years
        
        return {
            "allocation": allocation,
            "metrics": {
                "Monthly Investment": investment_amount,
                "Expected Annual Return": total_return * 100,
                "Investment Period": years,
                "Projected Value": projected_value
            }
        }

    def plot_competitor_performance(self, competitor_data: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create a competitor performance comparison visualization"""
        if not competitor_data:
            # Create empty chart with message if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No competitor data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig
        
        # Prepare data for plotting
        companies = list(competitor_data.keys())
        metrics = ['Revenue Growth', 'Profit Margin', 'Market Share']
        
        # Create subplots
        fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)
        
        # Add bars for each metric
        for i, metric in enumerate(metrics, 1):
            values = [data[metric] for data in competitor_data.values()]
            colors = ['#00CC96' if v >= 0 else '#EF553B' for v in values]
            
            fig.add_trace(
                go.Bar(
                    x=companies,
                    y=values,
                    name=metric,
                    marker_color=colors,
                    text=[f"{v:+.1f}%" for v in values],
                    textposition='auto',
                ),
                row=1, col=i
            )
        
        # Update layout
        fig.update_layout(
            height=400,
            showlegend=False,
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_text="Competitor Analysis",
            title_x=0.5
        )
        
        # Update axes
        for i in range(1, 4):
            fig.update_xaxes(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickangle=45,
                row=1, col=i
            )
            fig.update_yaxes(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.2)',
                row=1, col=i
            )
        
        return fig

    def calculate_macd(self, data):
        """Calculate MACD indicator"""
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal
        }, index=data.index)
    
    def calculate_rsi(self, data, periods=14):
        """Calculate RSI indicator"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return pd.DataFrame({'RSI': rsi}, index=data.index)
    
    def calculate_moving_averages(self, data):
        """Calculate moving averages"""
        df = data.copy()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        return df
    
    def calculate_bollinger_bands(self, data, window=20):
        """Calculate Bollinger Bands"""
        df = data.copy()
        df['MA'] = df['Close'].rolling(window=window).mean()
        df['STD'] = df['Close'].rolling(window=window).std()
        df['Upper'] = df['MA'] + (df['STD'] * 2)
        df['Lower'] = df['MA'] - (df['STD'] * 2)
        return df
    
    def generate_trading_signals(self, data):
        """Generate trading signals based on technical indicators"""
        # Calculate indicators
        macd = self.calculate_macd(data)
        rsi = self.calculate_rsi(data)
        ma = self.calculate_moving_averages(data)
        bb = self.calculate_bollinger_bands(data)
        
        # Get latest values
        latest_macd = macd['MACD'].iloc[-1]
        latest_signal = macd['Signal'].iloc[-1]
        latest_rsi = rsi['RSI'].iloc[-1]
        latest_price = data['Close'].iloc[-1]
        latest_sma20 = ma['SMA_20'].iloc[-1]
        latest_sma50 = ma['SMA_50'].iloc[-1]
        latest_bb_upper = bb['Upper'].iloc[-1]
        latest_bb_lower = bb['Lower'].iloc[-1]
        
        # Determine trend
        trend = 'Bullish' if latest_sma20 > latest_sma50 else 'Bearish'
        
        # Calculate signal strength (0-100)
        strength = 0
        
        # MACD crossover (30 points)
        if latest_macd > latest_signal:
            strength += 30
        
        # RSI conditions (40 points)
        if latest_rsi < 30:  # Oversold
            strength += 40
        elif latest_rsi > 70:  # Overbought
            strength += 0
        else:  # Neutral
            strength += 20
        
        # Bollinger Bands (30 points)
        if latest_price < latest_bb_lower:
            strength += 30
        elif latest_price > latest_bb_upper:
            strength += 0
        else:
            strength += 15
        
        # Determine action
        if strength >= 70:
            action = 'Buy'
        elif strength <= 30:
            action = 'Sell'
        else:
            action = 'Hold'
        
        return {
            'Trend': trend,
            'Signal_Strength': strength,
            'Recommended_Action': action
        }
    
    def generate_insights(self, data, predictions):
        """Generate insights based on technical analysis and predictions"""
        insights = {
            'Price Trends': [],
            'Technical Signals': [],
            'Prediction Analysis': [],
            'Risk Assessment': []
        }
        
        # Price Trends
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        if price_change > 0:
            insights['Price Trends'].append(f"Stock is up {price_change:.2f}% from previous close")
        else:
            insights['Price Trends'].append(f"Stock is down {abs(price_change):.2f}% from previous close")
        
        # Technical Signals
        rsi = self.calculate_rsi(data)['RSI'].iloc[-1]
        if rsi > 70:
            insights['Technical Signals'].append(f"RSI at {rsi:.2f} indicates overbought conditions")
        elif rsi < 30:
            insights['Technical Signals'].append(f"RSI at {rsi:.2f} indicates oversold conditions")
        
        macd = self.calculate_macd(data)
        if macd['MACD'].iloc[-1] > macd['Signal'].iloc[-1]:
            insights['Technical Signals'].append("MACD shows bullish crossover")
        elif macd['MACD'].iloc[-1] < macd['Signal'].iloc[-1]:
            insights['Technical Signals'].append("MACD shows bearish crossover")
        
        # Prediction Analysis
        if 'Predicted_Price' in predictions.columns:
            last_actual = predictions['Actual_Price'].iloc[-1]
            next_pred = predictions['Predicted_Price'].iloc[-1]
            pred_change = ((next_pred - last_actual) / last_actual) * 100
            
            if pred_change > 0:
                insights['Prediction Analysis'].append(f"Model predicts a {pred_change:.2f}% increase")
            else:
                insights['Prediction Analysis'].append(f"Model predicts a {abs(pred_change):.2f}% decrease")
        
        # Risk Assessment
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
        insights['Risk Assessment'].append(f"Annual volatility is {volatility:.2f}%")
        
        bb = self.calculate_bollinger_bands(data)
        if data['Close'].iloc[-1] > bb['Upper'].iloc[-1]:
            insights['Risk Assessment'].append("Price is above upper Bollinger Band, suggesting increased downside risk")
        elif data['Close'].iloc[-1] < bb['Lower'].iloc[-1]:
            insights['Risk Assessment'].append("Price is below lower Bollinger Band, suggesting increased upside potential")
        
        return insights