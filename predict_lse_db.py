import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
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
    """Get stock data with exchange-specific handling and feature engineering"""
    try:
        # Ensure LSE suffix is present
        if not symbol.endswith('.L'):
            symbol = f"{symbol}.L"
            
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
        
        # Get fundamental features with proper validation
        info = stock.info
        
        # Get latest data point
        latest = hist.iloc[-1]
        
        # Handle P/E ratio - use None if negative or invalid
        pe_forward = info.get('forwardPE')
        pe_trailing = info.get('trailingPE')
        pe_ratio = None
        if pe_forward is not None and pe_forward > 0:
            pe_ratio = pe_forward
        elif pe_trailing is not None and pe_trailing > 0:
            pe_ratio = pe_trailing
            
        # Handle P/B ratio - use None if negative or invalid
        pb_ratio = info.get('priceToBook')
        if pb_ratio is not None and pb_ratio < 0:
            pb_ratio = None
            
        # Handle other metrics
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        market_cap = info.get('marketCap', 0) if info.get('marketCap') else 0
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity is not None and debt_to_equity < 0:
            debt_to_equity = None
        
        # Get all required features
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
        
        # Create feature matrix with exactly 16 features as expected by the model
        # Replace None values with 0 for model input
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
            pe_ratio if pe_ratio is not None else 0,        # PE_Ratio
            pb_ratio if pb_ratio is not None else 0,        # PB_Ratio
            dividend_yield,   # Dividend_Yield
            market_cap,      # Market_Cap
            debt_to_equity if debt_to_equity is not None else 0   # Debt_To_Equity
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
        
        # Get additional fundamental data from stock object
        stock = data['stock']
        info = stock.info
        current_ratio = info.get('currentRatio', None)
        profit_margin = info.get('profitMargin', None)
        if profit_margin:
            profit_margin *= 100  # Convert to percentage
        operating_margin = info.get('operatingMargin', None)
        if operating_margin:
            operating_margin *= 100  # Convert to percentage
        roe = info.get('returnOnEquity', None)
        if roe:
            roe *= 100  # Convert to percentage
        revenue_growth = info.get('revenueGrowth', None)
        if revenue_growth:
            revenue_growth *= 100  # Convert to percentage
        earnings_growth = info.get('earningsGrowth', None)
        if earnings_growth:
            earnings_growth *= 100  # Convert to percentage
        free_cash_flow = info.get('freeCashflow', None)
        
        # Prepare data for Supabase
        prediction_data = {
            # Basic Information
            'symbol': symbol,
            'company_name': data['company_name'],
            'sector': data['sector'],
            'industry': data['industry'],
            
            # Price and Prediction
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'confidence_interval_lower': float(ci_lower),
            'confidence_interval_upper': float(ci_upper),
            'probability': float(prob),
            'rating': rating,
            'expected_return': float(expected_return),
            
            # Technical Indicators
            'sma_20': float(data['features'][0][1]),  # SMA20
            'sma_50': float(data['features'][0][2]),  # SMA50
            'rsi': float(data['features'][0][8]),     # RSI
            'macd': float(data['features'][0][6]),    # MACD
            'macd_signal': float(data['features'][0][7]),  # MACD Signal
            'bb_upper': float(data['features'][0][3]),    # BB Upper
            'bb_lower': float(data['features'][0][4]),    # BB Lower
            'bb_middle': float(data['features'][0][5]),   # BB Middle
            'force_index': float(data['features'][0][9]), # Force Index
            'volume_ratio': float(data['features'][0][10]), # Volume Ratio
            'daily_returns': float(data['features'][0][0]),  # Returns
            'volume': int(data['volume']),
            'volatility': float(volatility),
            
            # Fundamental Data
            'market_cap': int(data['features'][0][14]) if data['features'][0][14] > 0 else None,  # Market Cap
            'pe_ratio': float(data['features'][0][11]) if data['features'][0][11] > 0 else None,  # PE Ratio
            'pb_ratio': float(data['features'][0][12]) if data['features'][0][12] > 0 else None,  # PB Ratio
            'dividend_yield': float(data['features'][0][13]) if data['features'][0][13] >= 0 else 0,  # Dividend Yield
            'debt_to_equity': float(data['features'][0][15]) if data['features'][0][15] >= 0 else None,  # Debt to Equity
            'current_ratio': current_ratio,
            'profit_margin': profit_margin,
            'operating_margin': operating_margin,
            'return_on_equity': roe,
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth,
            'free_cash_flow': free_cash_flow,
            
            # Metadata
            'prediction_date': datetime.now().isoformat()
        }
        
        # Store prediction in Supabase
        result = supabase.table('lse_stock_data').insert(prediction_data).execute()
        
        # Print results
        print(f"\nPrediction for {symbol}:")
        print(f"Rating: {rating}")
        print(f"Probability: {prob:.2f}%")
        print(f"Current Price: £{current_price:.2f}")
        print(f"Tomorrow's Predicted Price: £{predicted_price:.2f}")
        print(f"95% Confidence Interval: £{ci_lower:.2f} - £{ci_upper:.2f}")
        print(f"Volume: {int(data['volume']):,}")
        print("Prediction stored in database")
        
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
    
    # Load LSE symbols from CSV
    lse_file = 'lse.csv'
    if not os.path.exists(lse_file):
        raise FileNotFoundError(f"LSE symbols file not found: {lse_file}")
        
    print("\nLoading LSE symbols...")
    lse_df = pd.read_csv(lse_file)
    test_stocks = lse_df['Symbol'].tolist()
    print(f"Loaded {len(test_stocks)} LSE symbols")
    
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
    
    print("\nMaking stock predictions and storing in database...")
    print("=" * 50)
    
    for stock in test_stocks:
        predict_stock(stock, model)
        print("-" * 50)
