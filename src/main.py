from ml_predictor import StockPredictor
import pandas as pd
import plotly.io as pio

def main():
    # Create an instance of StockPredictor
    predictor = StockPredictor()
    
    # Example: Predict AAPL stock prices for next 14 days
    symbol = "AAPL"  # Apple Inc.
    days = 14
    
    print(f"\nPredicting stock prices for {symbol} for the next {days} days...")
    
    # Get predictions
    predictions = predictor.predict_future_prices(symbol, days)
    
    if predictions:
        print("\nPrediction Results:")
        print(f"Trend: {predictions['trend']}")
        print(f"Confidence: {predictions['confidence']:.2f}%")
        print(f"\nPredicted Price Range:")
        print(f"Min: ${predictions['price_range'][0]:.2f}")
        print(f"Max: ${predictions['price_range'][1]:.2f}")
        
        print(f"\nRisk Level: {predictions['risk_level']}")
        print(f"Volatility: {predictions['volatility']:.2f}%")
        
        print("\nTechnical Signals:")
        for signal, value in predictions['technical_signals'].items():
            print(f"{signal}: {value}")
        
        print("\nMarket Sentiment:")
        for key, value in predictions['market_sentiment'].items():
            print(f"{key}: {value}")
        
        # Save the prediction chart
        fig = predictions['chart']
        pio.write_html(fig, 'stock_prediction.html')
        print("\nPrediction chart has been saved to 'stock_prediction.html'")
        
        # Get model performance metrics
        performance = predictor.get_model_performance_table(pd.DataFrame())
        if performance:
            print("\nModel Performance Metrics:")
            print(performance['table'])
    else:
        print("Failed to generate predictions.")

if __name__ == "__main__":
    main() 