import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import yfinance as yf
import ta
from ta.trend import ADXIndicator, IchimokuIndicator
from ta.volume import ChaikinMoneyFlowIndicator
import requests
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 60
        self.feature_columns = None
        tf.keras.backend.clear_session()
        self.model = None
        
    def _build_model(self, n_features):
        """Build and compile the LSTM model"""
        try:
            self.model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(self.sequence_length, n_features)),
                Dropout(0.2),
                LSTM(100, return_sequences=True),
                Dropout(0.2),
                LSTM(100),
                Dropout(0.2),
                Dense(50, activation='relu'),
                Dense(1)
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
        except Exception as e:
            print(f"Error building model: {str(e)}")
            raise e

    def _get_fundamental_metrics(self, symbol):
        """Get fundamental metrics for a stock"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Basic fundamentals
            metrics = {
                'FCF': info.get('freeCashflow', 0),
                'BVPS': info.get('bookValue', 0),
                'NetProfitMargin': info.get('profitMargins', 0),
                'EarningsGrowth': info.get('earningsQuarterlyGrowth', 0),
                'Beta': info.get('beta', 1),
                'PE': info.get('trailingPE', 0),
                'PB': info.get('priceToBook', 0),
                'DebtToEquity': info.get('debtToEquity', 0),
                'ROE': info.get('returnOnEquity', 0),
                'ROA': info.get('returnOnAssets', 0)
            }
            
            return metrics
        except Exception as e:
            print(f"Error getting fundamental metrics: {str(e)}")
            return {}

    def _get_sector_metrics(self, symbol):
        """Get sector-specific metrics"""
        try:
            stock = yf.Ticker(symbol)
            sector = stock.info.get('sector', '')
            industry = stock.info.get('industry', '')
            
            # Get sector ETF data (example: XLF for Financial sector)
            sector_etfs = {
                'Financial': 'XLF',
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Consumer Cyclical': 'XLY',
                'Energy': 'XLE'
            }
            
            sector_etf = sector_etfs.get(sector)
            if sector_etf:
                etf_data = yf.download(sector_etf, period='1y')
                sector_performance = (etf_data['Close'].pct_change().mean() * 252) * 100
            else:
                sector_performance = 0
                
            return {
                'SectorPerformance': sector_performance,
                'Sector': sector,
                'Industry': industry
            }
        except Exception as e:
            print(f"Error getting sector metrics: {str(e)}")
            return {}

    def _add_advanced_technical_indicators(self, df):
        """Add advanced technical indicators"""
        try:
            # ADX
            adx = ADXIndicator(df['High'], df['Low'], df['Close'])
            df['ADX'] = adx.adx()
            
            # Ichimoku Cloud
            ichimoku = IchimokuIndicator(df['High'], df['Low'])
            df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
            df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
            
            # Chaikin Money Flow
            cmf = ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'])
            df['CMF'] = cmf.chaikin_money_flow()
            
            # Fibonacci Retracement
            high = df['High'].max()
            low = df['Low'].min()
            diff = high - low
            df['Fib_38.2'] = high - (diff * 0.382)
            df['Fib_50.0'] = high - (diff * 0.5)
            df['Fib_61.8'] = high - (diff * 0.618)
            
            return df
        except Exception as e:
            print(f"Error adding advanced technical indicators: {str(e)}")
            return df

    def prepare_data(self, data, symbol=None):
        """Prepare data for training/prediction"""
        try:
            # Calculate technical indicators
            df = self._add_technical_indicators(data)
            df = self._add_advanced_technical_indicators(df)
            print(f"Shape after adding indicators: {df.shape}")
            
            # Add fundamental metrics if symbol is provided
            if symbol:
                fund_metrics = self._get_fundamental_metrics(symbol)
                sector_metrics = self._get_sector_metrics(symbol)
                
                for key, value in {**fund_metrics, **sector_metrics}.items():
                    df[key] = value
            
            # Select features
            self.feature_columns = [
                # Price and Volume
                'Open', 'High', 'Low', 'Close', 'Volume',
                # Basic Technical Indicators
                'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
                'BB_High', 'BB_Low', 'Volatility', 'ROC',
                # Advanced Technical Indicators
                'ADX', 'CMF', 'Ichimoku_Conversion', 'Ichimoku_Base',
                'Fib_38.2', 'Fib_50.0', 'Fib_61.8'
            ]
            
            # Add fundamental metrics if available
            if symbol:
                self.feature_columns.extend([
                    'FCF', 'BVPS', 'NetProfitMargin', 'EarningsGrowth',
                    'Beta', 'PE', 'PB', 'DebtToEquity', 'ROE', 'ROA',
                    'SectorPerformance'
                ])
            
            # Ensure all required columns exist
            for col in self.feature_columns:
                if col not in df.columns:
                    print(f"Missing column: {col}")
                    df[col] = 0
            
            # Select only the required features
            df = df[self.feature_columns]
            print(f"Selected features shape: {df.shape}")
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(df)
            print(f"Scaled data shape: {scaled_data.shape}")
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:(i + self.sequence_length)])
                y.append(scaled_data[i + self.sequence_length, df.columns.get_loc('Close')])
            
            X = np.array(X)
            y = np.array(y)
            print(f"Final shapes - X: {X.shape}, y: {y.shape}")
            
            return X, y, len(self.feature_columns)
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            raise e

    def _add_technical_indicators(self, df):
        """Add basic technical indicators"""
        try:
            df = df.copy()
            
            # Basic indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_High'] = bollinger.bollinger_hband()
            df['BB_Low'] = bollinger.bollinger_lband()
            
            # Volatility
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
            
            # Rate of Change
            df['ROC'] = df['Close'].pct_change(periods=20)
            
            # Fill NaN values
            df = df.ffill().bfill()
            
            return df
        except Exception as e:
            print(f"Error adding technical indicators: {str(e)}")
            raise e

    def train(self, data, symbol=None):
        """Train the model on historical data"""
        try:
            # Prepare data and build model with correct number of features
            X, y, n_features = self.prepare_data(data, symbol)
            if self.model is None:
                self._build_model(n_features)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                verbose=1
            )
            return history
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise e

    def predict(self, data, symbol=None, days_ahead=30):
        """Make predictions for future stock prices"""
        try:
            if self.model is None:
                print("Training model first...")
                self.train(data, symbol)
            
            # Prepare the last sequence from the data
            X, _, n_features = self.prepare_data(data, symbol)
            last_sequence = X[-1:]
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            # Predict for the specified number of days
            for _ in range(days_ahead):
                pred = self.model.predict(current_sequence, verbose=0)[0][0]
                predictions.append(pred)
                
                # Update the sequence for next prediction
                new_sequence = np.roll(current_sequence[0], -1, axis=0)
                new_sequence[-1] = pred
                current_sequence = new_sequence.reshape(1, self.sequence_length, n_features)
            
            # Inverse transform predictions
            close_price_scaler = MinMaxScaler()
            close_prices = data['Close'].values.reshape(-1, 1)
            close_price_scaler.fit(close_prices)
            
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = close_price_scaler.inverse_transform(predictions)
            
            return predictions.flatten()
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            raise e

    def sentiment_analysis(self, news_data):
        """Analyze sentiment from news data"""
        try:
            # Placeholder for sentiment analysis
            # This will be implemented with BERT or similar models
            pass
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            raise e

# Example usage
# predictor = StockPredictor()
# predictor.train(X_train, y_train)
# predictions = predictor.predict(X_test)