import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from stock_model import StockPredictor

# Initialize scaler for feature preprocessing
scaler = StandardScaler()

def get_stock_data(symbol):
    """Get stock data with exchange-specific handling and feature engineering
    Matches the model's training feature set:
    - Technical features: RSI, BB, MACD, Force Index, Volume Ratio
    - Fundamental features: Company metrics, Valuation metrics, Financial health
    """
    try:
        # Download data
        stock = yf.Ticker(symbol)
        hist = stock.history(period='1y')
        
        if len(hist) < 50:
            return None
        
        # Calculate technical indicators
        hist['Returns'] = hist['Close'].pct_change()
        hist['SMA20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        
        # Bollinger Bands
        hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
        bb_std = hist['Close'].rolling(window=20).std()
        hist['BB_Upper'] = hist['BB_Middle'] + (bb_std * 2)
        hist['BB_Lower'] = hist['BB_Middle'] - (bb_std * 2)
        
        # MACD
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = exp1 - exp2
        hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Force Index and Volume Ratio
        hist['Force_Index'] = hist['Close'].diff(1) * hist['Volume']
        hist['Volume_Ratio'] = hist['Volume'] / hist['Volume'].rolling(window=20).mean()
        
        # Get fundamental features
        info = stock.info
        
        # Get latest data point
        latest = hist.iloc[-1]
        
        # Get all required features based on model's training
        returns = latest['Returns']
        sma20 = latest['SMA20']
        sma50 = latest['SMA50']
        bb_upper = latest['BB_Upper']
        bb_lower = latest['BB_Lower']
        bb_middle = latest['BB_Middle']
        macd = latest['MACD']
        signal_line = latest['Signal_Line']
        rsi = latest['RSI']
        force_index = latest['Force_Index']
        volume_ratio = latest['Volume_Ratio']
        
        # Get fundamental features
        pe_ratio = info.get('forwardPE', info.get('trailingPE', 0))
        pb_ratio = info.get('priceToBook', 0)
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        market_cap = info.get('marketCap', 0)
        debt_to_equity = info.get('debtToEquity', 0)
        
        # Create feature matrix with exactly 16 features as expected by the model
        feature_matrix = np.array([
            returns,          # Returns
            sma20,           # SMA20
            sma50,           # SMA50
            bb_upper,        # BB_Upper
            bb_lower,        # BB_Lower
            bb_middle,       # BB_Middle
            macd,            # MACD
            signal_line,     # Signal_Line
            rsi,             # RSI
            force_index,     # Force_Index
            volume_ratio,    # Volume_Ratio
            pe_ratio,        # PE_Ratio
            pb_ratio,        # PB_Ratio
            dividend_yield,   # Dividend_Yield
            market_cap,      # Market_Cap
            debt_to_equity   # Debt_To_Equity
        ]).reshape(1, -1)
        
        return {
            'features': feature_matrix,
            'last_price': latest['Close'],
            'volume': latest['Volume'],
            'history': hist  # Return historical data for volatility calculation
        }
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None

def predict_stock(symbol, model):
    """Make prediction for a single stock"""
    try:
        # Get stock data
        data = get_stock_data(symbol)
        if not data:
            print(f"Could not get data for {symbol}")
            return
        
        # Get features
        features = data['features']
        
        # 1. Handle missing values and infinities
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Scale features using pre-fitted scaler
        features = scaler.transform(features)
        
        # 3. Make prediction using the VotingClassifier
        prob = model.predict_proba(features)[0][1] * 100  # Convert to percentage
        
        # 4. Apply rating system based on probability thresholds
        if prob >= 80:
            rating = "Strong Buy"
            expected_return = 0.08  # 8% expected return
        elif prob >= 65:
            rating = "Buy"
            expected_return = 0.05  # 5% expected return
        elif prob > 35:
            rating = "Weak Buy"
            expected_return = 0.02  # 2% expected return
        elif prob <= 20:
            rating = "Strong Sell"
            expected_return = -0.08  # -8% expected return
        elif prob <= 35:
            rating = "Sell"
            expected_return = -0.05  # -5% expected return
        else:
            rating = "Weak Sell"
            expected_return = -0.02  # -2% expected return
            
        # Calculate tomorrow's predicted price
        current_price = data['last_price']
        predicted_price = current_price * (1 + expected_return)
        
        # Calculate volatility for confidence interval
        returns = data['history']['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        daily_volatility = volatility / np.sqrt(252)
        
        # Calculate confidence intervals based on probability strength
        prob_strength = abs(prob - 50) / 50  # 0 to 1 scale of prediction strength
        confidence_multiplier = 2.0 - prob_strength  # More confident = narrower interval
        ci_width = 1.96 * daily_volatility * confidence_multiplier
        
        ci_lower = predicted_price * (1 - ci_width)
        ci_upper = predicted_price * (1 + ci_width)
        
        # Print results
        print(f"\nPrediction for {symbol}:")
        print(f"Rating: {rating}")
        print(f"Probability: {prob:.2f}%")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Tomorrow's Predicted Price: ${predicted_price:.2f}")
        print(f"95% Confidence Interval: ${ci_lower:.2f} - ${ci_upper:.2f}")
        print(f"Volume: {int(data['volume']):,}")
        
    except Exception as e:
        print(f"Error predicting {symbol}: {str(e)}")

if __name__ == "__main__":
    # Load the pre-trained model
    print("Loading pre-trained model...")
    model_path = os.path.join('trained_models', 'ensemble_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found at {model_path}")
    print(f"Loading pre-trained model from {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    
    # Load NASDAQ symbols from CSV
    nasdaq_file = 'nasdaq.csv'
    if not os.path.exists(nasdaq_file):
        raise FileNotFoundError(f"NASDAQ symbols file not found: {nasdaq_file}")
        
    print("\nLoading NASDAQ symbols...")
    nasdaq_df = pd.read_csv(nasdaq_file)
    test_stocks = nasdaq_df['Symbol'].tolist()
    print(f"Loaded {len(test_stocks)} NASDAQ symbols")
    
    # First collect all features to fit the scaler
    print("\nCollecting features for scaling...")
    all_features = []
    for symbol in test_stocks:
        data = get_stock_data(symbol)
        if data is not None:
            all_features.append(data['features'])
            
    if not all_features:
        print("Error: Could not collect features from any stock")
        exit(1)
        
    # Fit scaler on all features
    all_features = np.vstack(all_features)
    scaler.fit(all_features)
    
    print("\nMaking stock predictions...")
    print("=" * 50)
    
    for stock in test_stocks:
        predict_stock(stock, model)
        print("-" * 50)
