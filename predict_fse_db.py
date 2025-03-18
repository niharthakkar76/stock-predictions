import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from stock_model import StockPredictor
from supabase import create_client, Client

# Initialize Supabase client
SUPABASE_URL = "https://dcitsfjssvlgtqfqrriq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRjaXRzZmpzc3ZsZ3RxZnFycmlxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MjEyMDE1MSwiZXhwIjoyMDU3Njk2MTUxfQ.PtHlriHLSyrfxdvSvprlL4S5OulQBlv-XKOJmoB659o"

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize scaler for feature preprocessing
scaler = StandardScaler()

def get_stock_data(symbol):
    """Get stock data with FSE-specific handling"""
    try:
        # Ensure FSE suffix is present
        if not symbol.endswith('.DE'):
            symbol = f"{symbol}.DE"
            
        # Download data with retry logic for FSE stocks
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1y')
            if len(hist) == 0:
                # Try alternative symbol format without .DE
                base_symbol = symbol.replace('.DE', '')
                stock = yf.Ticker(base_symbol)
                hist = stock.history(period='1y')
        except Exception as e:
            print(f"${symbol}: {str(e)}")
            return None
        
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
        
        # Get latest data point
        latest = hist.iloc[-1]
        info = stock.info
        
        # Handle P/E ratio
        pe_forward = info.get('forwardPE')
        pe_trailing = info.get('trailingPE')
        pe_ratio = None
        if pe_forward is not None and pe_forward > 0:
            pe_ratio = pe_forward
        elif pe_trailing is not None and pe_trailing > 0:
            pe_ratio = pe_trailing
            
        # Handle P/B ratio
        pb_ratio = info.get('priceToBook')
        if pb_ratio is not None and pb_ratio < 0:
            pb_ratio = None
            
        # Handle other metrics
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        market_cap = info.get('marketCap', 0) if info.get('marketCap') else 0
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity is not None and debt_to_equity < 0:
            debt_to_equity = None
        
        # Handle extreme values before creating feature matrix
        def safe_value(val, max_val=1e6):
            if val is None or np.isnan(val) or np.isinf(val):
                return 0
            return max(min(float(val), max_val), -max_val)
            
        # Create feature matrix with safe values
        feature_matrix = np.array([
            safe_value(latest['Returns']),
            safe_value(latest['SMA20']),
            safe_value(latest['SMA50']),
            safe_value(latest['BB_Upper']),
            safe_value(latest['BB_Lower']),
            safe_value(latest['BB_Middle']),
            safe_value(latest['MACD']),
            safe_value(latest['Signal_Line']),
            safe_value(latest['RSI']),
            safe_value(latest['Force_Index']),
            safe_value(latest['Volume_Ratio']),
            pe_ratio if pe_ratio is not None else 0,
            pb_ratio if pb_ratio is not None else 0,
            dividend_yield,
            market_cap,
            debt_to_equity if debt_to_equity is not None else 0
        ]).reshape(1, -1)
        
        return {
            'features': feature_matrix,
            'last_price': latest['Close'],
            'volume': latest['Volume'],
            'history': hist,
            'company_name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'stock': stock
        }
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None

def predict_stock(symbol, model):
    """Make prediction for a single stock and store in Supabase"""
    try:
        data = get_stock_data(symbol)
        if not data:
            print(f"Could not get data for {symbol}")
            return
        
        # Handle missing values and scale features
        features = np.nan_to_num(data['features'], nan=0.0, posinf=0.0, neginf=0.0)
        features = scaler.transform(features)
        
        # Make prediction
        prob = model.predict_proba(features)[0][1] * 100
        
        # Apply rating system
        if prob >= 80:
            rating = "Strong Buy"
            expected_return = 0.08
        elif prob >= 65:
            rating = "Buy"
            expected_return = 0.05
        elif prob > 35:
            rating = "Weak Buy"
            expected_return = 0.02
        elif prob <= 20:
            rating = "Strong Sell"
            expected_return = -0.08
        elif prob <= 35:
            rating = "Sell"
            expected_return = -0.05
        else:
            rating = "Weak Sell"
            expected_return = -0.02
            
        # Calculate predictions
        current_price = data['last_price']
        predicted_price = current_price * (1 + expected_return)
        
        # Calculate confidence interval
        returns = data['history']['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        daily_volatility = volatility / np.sqrt(252)
        
        prob_strength = abs(prob - 50) / 50
        confidence_multiplier = 2.0 - prob_strength
        ci_width = 1.96 * daily_volatility * confidence_multiplier
        
        ci_lower = predicted_price * (1 - ci_width)
        ci_upper = predicted_price * (1 + ci_width)
        
        # Get additional metrics
        stock = data['stock']
        info = stock.info
        
        # Store prediction in Supabase
        prediction_data = {
            'symbol': symbol,
            'company_name': data['company_name'],
            'sector': data['sector'],
            'industry': data['industry'],
            
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'confidence_interval_lower': float(ci_lower),
            'confidence_interval_upper': float(ci_upper),
            'probability': min(float(prob), 99.99),  # Ensure within decimal(5,2) range
            'rating': rating,
            'expected_return': max(min(float(expected_return), 99.99), -99.99),
            
            'sma_20': float(data['features'][0][1]),
            'sma_50': float(data['features'][0][2]),
            'rsi': float(data['features'][0][8]),
            'macd': float(data['features'][0][6]),
            'macd_signal': float(data['features'][0][7]),
            'bb_upper': float(data['features'][0][3]),
            'bb_lower': float(data['features'][0][4]),
            'bb_middle': float(data['features'][0][5]),
            'force_index': float(data['features'][0][9]),
            'volume_ratio': float(data['features'][0][10]),
            'daily_returns': float(data['features'][0][0]),
            'volume': int(data['volume']),
            'volatility': float(volatility),
            
            'market_cap': int(data['features'][0][14]) if data['features'][0][14] > 0 else None,
            'pe_ratio': min(float(data['features'][0][11]), 999.99) if data['features'][0][11] > 0 else None,
            'pb_ratio': min(float(data['features'][0][12]), 999.99) if data['features'][0][12] > 0 else None,
            'dividend_yield': min(float(data['features'][0][13]), 99.99) if data['features'][0][13] >= 0 else 0,
            'debt_to_equity': min(float(data['features'][0][15]), 999.99) if data['features'][0][15] >= 0 else None,
            'current_ratio': info.get('currentRatio'),
            'profit_margin': min(info.get('profitMargin', 0) * 100, 999.99) if info.get('profitMargin') else None,
            'operating_margin': min(info.get('operatingMargin', 0) * 100, 999.99) if info.get('operatingMargin') else None,
            'return_on_equity': min(info.get('returnOnEquity', 0) * 100, 999.99) if info.get('returnOnEquity') else None,
            'revenue_growth': min(info.get('revenueGrowth', 0) * 100, 999.99) if info.get('revenueGrowth') else None,
            'earnings_growth': min(info.get('earningsGrowth', 0) * 100, 999.99) if info.get('earningsGrowth') else None,
            'free_cash_flow': info.get('freeCashflow'),
            
            'prediction_date': datetime.now().isoformat()
        }
        
        result = supabase.table('fse_stock_data').insert(prediction_data).execute()
        
        print(f"\nPrediction for {symbol}:")
        print(f"Rating: {rating}")
        print(f"Probability: {prob:.2f}%")
        print(f"Current Price: €{current_price:.2f}")
        print(f"Tomorrow's Predicted Price: €{predicted_price:.2f}")
        print(f"95% Confidence Interval: €{ci_lower:.2f} - €{ci_upper:.2f}")
        print(f"Volume: {int(data['volume']):,}")
        print("Prediction stored in database")
        
    except Exception as e:
        print(f"Error predicting {symbol}: {str(e)}")

if __name__ == "__main__":
    print("Loading pre-trained model...")
    model_path = os.path.join('trained_models', 'ensemble_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found at {model_path}")
    print(f"Loading pre-trained model from {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    
    print("\nLoading FSE symbols...")
    fse_df = pd.read_csv('fse.csv')
    test_stocks = fse_df['Symbol'].tolist()
    print(f"Loaded {len(test_stocks)} FSE symbols")
    
    print("\nCollecting features for scaling...")
    all_features = []
    for symbol in test_stocks:
        data = get_stock_data(symbol)
        if data is not None:
            all_features.append(data['features'])
            
    if not all_features:
        print("Error: Could not collect features from any stock")
        exit(1)
        
    all_features = np.vstack(all_features)
    scaler.fit(all_features)
    
    print("\nMaking stock predictions and storing in database...")
    print("=" * 50)
    
    for stock in test_stocks:
        predict_stock(stock, model)
        print("-" * 50)
