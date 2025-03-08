import yfinance as yf
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime, timedelta
from data_preprocessor import StockDataPreprocessor
import talib
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        # Load trained models
        self.rf_model = load('trained_models/rf_model.joblib')
        self.xgb_model = load('trained_models/xgb_model.joblib')
        self.gb_model = load('trained_models/gb_model.joblib')
        self.ensemble_model = load('trained_models/ensemble_model.joblib')
        self.preprocessor = StockDataPreprocessor()
        
        # Top 50 stocks to analyze as per project requirements
        self.stocks = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'ADBE', 'NFLX', 'CRM', 'INTC', 'CSCO',
            # Financial
            'JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK',
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABT', 'MRK', 'TMO',
            # Consumer
            'WMT', 'PG', 'KO', 'PEP', 'HD', 'MCD', 'NKE', 'SBUX',
            # Industrial
            'CAT', 'BA', 'HON', 'MMM', 'UPS', 'FDX',
            # Energy
            'XOM', 'CVX', 'COP',
            # Communication
            'VZ', 'T', 'CMCSA',
            # Others
            'DIS', 'NEE', 'RTX', 'LMT'
        ]

    def get_stock_data(self, symbol, days=90):
        """Get historical stock data with enough history for technical indicators"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)  # Get 90 days of data for better indicators
        
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date, interval='1d')
            if len(df) >= 50:  # Need at least 50 days for SMA50
                return df
            print(f"Warning: Insufficient data for {symbol} (only {len(df)} days)")
            return None
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def prepare_features(self, df):
        """Prepare features for prediction using the same features as training"""
        # Calculate the core technical indicators used in training
        df['SMA20'] = talib.SMA(df['Close'], timeperiod=20)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Volume force index
        df['Volume_Force'] = df['Volume'] * df['Returns']
        
        # Momentum
        df['Momentum'] = df['Close'].pct_change(5)
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Create final feature set matching training data
        feature_cols = [
            'Close',      # Price level
            'Volume',     # Trading volume
            'SMA20',      # Trend
            'RSI',        # Momentum
            'Volume_Force', # Volume & price combined
            'Momentum'    # Price momentum
        ]
        
        # Scale features using z-score normalization
        df_features = df[feature_cols].copy()
        df_features = (df_features - df_features.mean()) / df_features.std()
        
        return df_features

    def predict_stock(self, symbol):
        """Make predictions for a single stock"""
        print(f"Processing {symbol}...")
        
        # Get historical data
        df = self.get_stock_data(symbol)
        if df is None:
            return None
        
        try:
            # Prepare features
            df_processed = self.prepare_features(df)
            if len(df_processed) < 7:  # Need at least 7 days of data
                return None
            
            # Use the core features from training
            feature_cols = [
                'Close',      # Price level
                'Volume',     # Trading volume
                'SMA20',      # Trend
                'RSI',        # Momentum
                'Volume_Force', # Volume & price combined
                'Momentum'    # Price momentum
            ]
            
            # Ensure all required features are present
            missing_cols = set(feature_cols) - set(df_processed.columns)
            if missing_cols:
                print(f"Warning: Missing features for {symbol}: {missing_cols}")
                return None
            
            # Get last 7 days of data for prediction
            features = df_processed[feature_cols].iloc[-7:].values
            
            # Make predictions
            predictions = {
                'RF': self.rf_model.predict_proba(features)[:, 1],
                'XGB': self.xgb_model.predict_proba(features)[:, 1],
                'GB': self.gb_model.predict_proba(features)[:, 1],
                'Ensemble': self.ensemble_model.predict_proba(features)[:, 1]
            }
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            return None
        
        # Calculate average score
        avg_score = np.mean([predictions[model][-1] for model in predictions])
        
        # Get price movement
        price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        
        return {
            'symbol': symbol,
            'current_price': df['Close'].iloc[-1],
            'price_change': price_change,
            'prediction_score': avg_score,
            'future_predictions': predictions['Ensemble'],
            'confidence': np.std([predictions[model][-1] for model in predictions]) * -1 + 1  # Higher when models agree
        }

    def get_top_predictions(self, top_n=5):
        """Get predictions for top N stocks"""
        predictions = []
        
        print("\nAnalyzing stocks...")
        print("This may take a few minutes as we process technical indicators...\n")
        
        for symbol in self.stocks:
            result = self.predict_stock(symbol)
            if result:
                predictions.append(result)
                
        if not predictions:
            print("\nNo valid predictions could be generated. Please check data availability.")
            return []
        
        # Sort by prediction score and confidence
        predictions.sort(key=lambda x: (x['prediction_score'] * x['confidence']), reverse=True)
        
        return predictions[:top_n]

    def print_predictions(self, predictions):
        """Print formatted predictions with detailed analysis"""
        if not predictions:
            return
            
        print("\n" + "=" * 50)
        print("Top 5 Stock Predictions for", datetime.now().strftime('%Y-%m-%d'))
        print("=" * 50)
        print("\nRanked by prediction confidence and technical analysis")
        print("Predictions based on 90 days of historical data")
        print("Using ensemble of RF, XGB, and GB models\n")
        
        for pred in predictions:
            print(f"\nStock: {pred['symbol']}")
            print(f"Current Price: ${pred['current_price']:.2f}")
            print(f"24h Change: {pred['price_change']:.2f}%")
            
            # Color-coded prediction score
            score = pred['prediction_score']
            if score >= 0.7:
                strength = "Strong Buy"
            elif score >= 0.6:
                strength = "Buy"
            elif score >= 0.4:
                strength = "Hold"
            else:
                strength = "Neutral"
            
            print(f"Signal Strength: {strength}")
            print(f"Prediction Score: {score:.2%}")
            print(f"Model Confidence: {pred['confidence']:.2%}")
            
            print("\nNext 7 Days Prediction Trend:")
            scores = pred['future_predictions']
            
            # Get dates for next 7 days
            today = datetime.now()
            dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
            trend = 'Upward' if scores[-1] > scores[0] else 'Downward'
            volatility = np.std(scores)
            
            print(f"Trend Direction: {trend}")
            print(f"Trend Volatility: {volatility*100:.2f}%")
            print("\nDaily Predictions:")
            for date, score in zip(dates, scores):
                print(f"{date}: {score*100:.2f}%")
            
            print("-" * 50)

def main():
    predictor = StockPredictor()
    predictions = predictor.get_top_predictions()
    predictor.print_predictions(predictions)

if __name__ == "__main__":
    main()
