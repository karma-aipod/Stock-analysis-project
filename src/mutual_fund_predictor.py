import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

class MutualFundPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.rf_model = None
        self.gb_model = None
        self.feature_scaler = MinMaxScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        df = data.copy()
        
        # Keep only relevant features
        features = [
            'Close', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_20',
            'RSI', 'MACD_Line', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'ATR',
            'Daily_Return', 'Monthly_Return', 'Yearly_Return',
            'News_Sentiment'
        ]
        
        # Drop any rows with NaN values
        df = df[features].dropna()
        
        return df
    
    def prepare_sequences(self, data: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM model"""
        # Scale the features
        scaled_data = self.feature_scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, data.columns.get_loc('Close')])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: tuple) -> Sequential:
        """Build LSTM model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train_models(self, data: pd.DataFrame):
        """Train prediction models"""
        # Prepare features
        df = self.prepare_features(data)
        
        # Prepare data for LSTM
        X, y = self.prepare_sequences(df)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train LSTM
        self.lstm_model = self.build_lstm_model((X.shape[1], X.shape[2]))
        self.lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
        
        # Prepare data for traditional models
        features = df.drop(['Close'], axis=1)
        target = df['Close'].shift(-1).dropna()
        features = features[:-1]  # Remove last row to match target size
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(features, target)
        
        # Train Gradient Boosting
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.gb_model.fit(features, target)
    
    def predict_future_nav(self, data: pd.DataFrame, days: int = 14) -> Dict[str, Any]:
        """Predict future NAV prices"""
        try:
            print(f"Starting NAV prediction for {days} days")
            
            # Prepare features
            df = self.prepare_features(data)
            if df.empty:
                print("Error: No features could be prepared")
                return None
            print(f"Prepared features: {len(df)} rows with {len(df.columns)} columns")
            
            # Train models if not already trained
            if self.lstm_model is None:
                print("Training models...")
                try:
                    self.train_models(data)
                    print("Models trained successfully")
                except Exception as e:
                    print(f"Error training models: {str(e)}")
                    return None
            
            # Make predictions using ensemble approach
            try:
                print("Making LSTM predictions...")
                lstm_pred = self._predict_lstm(df, days)
                print("Making Random Forest predictions...")
                rf_pred = self._predict_traditional(df, self.rf_model, days)
                print("Making Gradient Boosting predictions...")
                gb_pred = self._predict_traditional(df, self.gb_model, days)
                
                # Verify predictions
                if len(lstm_pred) != days or len(rf_pred) != days or len(gb_pred) != days:
                    print("Error: Prediction lengths do not match requested days")
                    return None
                
                # Combine predictions (weighted average)
                ensemble_pred = (lstm_pred * 0.4 + rf_pred * 0.3 + gb_pred * 0.3)
                print("Ensemble predictions created")
                
                # Calculate metrics
                volatility = df['ATR'].mean() / df['Close'].mean() * 100
                trend = self._determine_trend(ensemble_pred)
                confidence = self._calculate_confidence(df)
                
                # Create prediction chart
                chart = self._create_prediction_chart(df, ensemble_pred)
                print("Chart created")
                
                result = {
                    'predictions': ensemble_pred,
                    'chart': chart,
                    'trend': trend,
                    'confidence': confidence,
                    'volatility': volatility,
                    'price_range': [
                        float(min(ensemble_pred)),
                        float(max(ensemble_pred))
                    ]
                }
                print("Prediction completed successfully")
                return result
                
            except Exception as e:
                print(f"Error during prediction process: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _predict_lstm(self, data: pd.DataFrame, days: int) -> np.ndarray:
        """Make predictions using LSTM model"""
        try:
            sequence = data.values[-60:]  # Last 60 days
            sequence = self.feature_scaler.transform(sequence)
            
            predictions = []
            current_sequence = sequence.copy()
            
            for _ in range(days):
                # Reshape for LSTM
                current_batch = current_sequence[-60:].reshape((1, 60, sequence.shape[1]))
                # Predict next value
                pred = self.lstm_model.predict(current_batch, verbose=0)[0]
                # Add prediction to sequence
                new_sequence = current_sequence[-59:].copy()
                new_row = current_sequence[-1].copy()
                new_row[data.columns.get_loc('Close')] = pred
                current_sequence = np.vstack([new_sequence, new_row])
                # Store the predicted close price
                predictions.append(pred)
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            scaled_predictions = np.zeros((len(predictions), data.shape[1]))
            scaled_predictions[:, data.columns.get_loc('Close')] = predictions[:, 0]
            predictions = self.feature_scaler.inverse_transform(scaled_predictions)[:, data.columns.get_loc('Close')]
            
            return predictions
            
        except Exception as e:
            print(f"Error in LSTM prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _predict_traditional(self, data: pd.DataFrame, model, days: int) -> np.ndarray:
        """Make predictions using traditional ML models"""
        try:
            features = data.drop(['Close'], axis=1)
            last_features = features.iloc[-1:]
            
            predictions = []
            current_features = last_features.copy()
            
            for _ in range(days):
                pred = model.predict(current_features)[0]
                predictions.append(pred)
                # Update features for next prediction
                current_features = self._update_features(current_features, pred)
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"Error in traditional model prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _update_features(self, features: pd.DataFrame, new_price: float) -> pd.DataFrame:
        """Update feature values for next prediction"""
        updated = features.copy()
        
        # Update technical indicators (simplified)
        for col in updated.columns:
            if col != 'News_Sentiment':
                updated[col] = updated[col].iloc[-1]
        
        return updated
    
    def _determine_trend(self, predictions: np.ndarray) -> str:
        """Determine trend direction from predictions"""
        if len(predictions) < 2:
            return "Neutral"
        
        start_price = predictions[0]
        end_price = predictions[-1]
        change_pct = (end_price - start_price) / start_price * 100
        
        if change_pct > 2:
            return "Strong Uptrend"
        elif change_pct > 0:
            return "Slight Uptrend"
        elif change_pct < -2:
            return "Strong Downtrend"
        elif change_pct < 0:
            return "Slight Downtrend"
        else:
            return "Neutral"
    
    def _calculate_confidence(self, data: pd.DataFrame) -> float:
        """Calculate prediction confidence score"""
        confidence = 70.0  # Base confidence
        
        # Adjust based on volatility
        volatility = data['ATR'].mean() / data['Close'].mean() * 100
        if volatility > 5:
            confidence -= 20
        elif volatility > 2:
            confidence -= 10
        
        # Adjust based on trend agreement
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        sma_200 = data['SMA_200'].iloc[-1]
        price = data['Close'].iloc[-1]
        
        if (price > sma_20 > sma_50 > sma_200) or (price < sma_20 < sma_50 < sma_200):
            confidence += 10  # Strong trend agreement
        
        # Adjust based on RSI
        rsi = data['RSI'].iloc[-1]
        if rsi > 70 or rsi < 30:
            confidence -= 10  # Overbought/Oversold conditions
        
        return min(max(confidence, 0), 100)
    
    def _create_prediction_chart(self, historical_data: pd.DataFrame, predictions: np.ndarray) -> go.Figure:
        """Create interactive chart with historical data and predictions"""
        # Create date range for predictions
        last_date = historical_data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions), freq='B')
        
        fig = go.Figure()
        
        # Historical NAV
        fig.add_trace(go.Scatter(
            x=historical_data.index[-90:],  # Last 90 days
            y=historical_data['Close'][-90:],
            name='Historical NAV',
            line=dict(color='#00CC96')
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            name='Predicted NAV',
            line=dict(color='#FFA15A', dash='dash')
        ))
        
        # Add confidence interval
        std = historical_data['Close'].std()
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions + std,
            fill=None,
            mode='lines',
            line_color='rgba(255,161,90,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions - std,
            fill='tonexty',
            mode='lines',
            line_color='rgba(255,161,90,0)',
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title='Mutual Fund NAV Prediction',
            xaxis_title='Date',
            yaxis_title='NAV',
            template='plotly_dark',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            )
        )
        
        return fig 