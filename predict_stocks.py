import numpy as np
import yfinance as yf
import joblib
import os
from datetime import datetime, timedelta
from stock_model import StockPredictor

def calculate_technical_indicators(prices, volumes):
    """Calculate technical indicators for the stock data"""
    # Calculate returns
    returns = np.zeros_like(prices)
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    
    # Calculate SMA
    sma20 = np.array([np.mean(prices[max(0, i-20+1):i+1]) for i in range(len(prices))])
    sma50 = np.array([np.mean(prices[max(0, i-50+1):i+1]) for i in range(len(prices))])
    
    # Calculate Bollinger Bands
    bb_middle = sma20
    rolling_std = np.array([np.std(prices[max(0, i-20+1):i+1]) for i in range(len(prices))])
    bb_upper = bb_middle + (rolling_std * 2)
    bb_lower = bb_middle - (rolling_std * 2)
    
    # Calculate MACD
    ema12 = np.zeros_like(prices)
    ema26 = np.zeros_like(prices)
    ema12[0] = prices[0]
    ema26[0] = prices[0]
    for i in range(1, len(prices)):
        ema12[i] = (prices[i] * 2/(12+1)) + (ema12[i-1] * (1 - 2/(12+1)))
        ema26[i] = (prices[i] * 2/(26+1)) + (ema26[i-1] * (1 - 2/(26+1)))
    macd = ema12 - ema26
    signal_line = np.zeros_like(macd)
    signal_line[0] = macd[0]
    for i in range(1, len(macd)):
        signal_line[i] = (macd[i] * 2/(9+1)) + (signal_line[i-1] * (1 - 2/(9+1)))
    
    # Calculate RSI
    deltas = np.diff(prices)
    gains = np.zeros_like(deltas)
    losses = np.zeros_like(deltas)
    gains[deltas > 0] = deltas[deltas > 0]
    losses[deltas < 0] = -deltas[deltas < 0]
    avg_gain = np.mean(gains[:14])
    avg_loss = np.mean(losses[:14])
    rsi = np.zeros_like(prices)
    if avg_loss == 0:
        rsi[14] = 100
    else:
        rs = avg_gain / avg_loss
        rsi[14] = 100 - (100 / (1 + rs))
    for i in range(15, len(prices)):
        avg_gain = ((avg_gain * 13) + gains[i-1]) / 14
        avg_loss = ((avg_loss * 13) + losses[i-1]) / 14
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return returns, sma20, sma50, bb_upper, bb_lower, bb_middle, macd, signal_line, rsi

def prepare_stock_features(data):
    """Prepare features for prediction using numpy arrays"""
    # Extract basic features
    closes = data[:, 4].astype(float)
    volumes = data[:, 5].astype(float)
    returns = np.zeros_like(closes)
    returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
    
    # Calculate SMA
    sma20 = np.array([np.mean(closes[max(0, i-20+1):i+1]) for i in range(len(closes))])
    sma50 = np.array([np.mean(closes[max(0, i-50+1):i+1]) for i in range(len(closes))])
    
    # Calculate RSI
    deltas = np.diff(closes)
    gains = np.zeros_like(deltas)
    losses = np.zeros_like(deltas)
    gains[deltas > 0] = deltas[deltas > 0]
    losses[deltas < 0] = -deltas[deltas < 0]
    
    avg_gain = np.mean(gains[:14])
    avg_loss = np.mean(losses[:14])
    rsi = np.zeros_like(closes)
    
    if avg_loss == 0:
        rsi[14] = 100
    else:
        rs = avg_gain / avg_loss
        rsi[14] = 100 - (100 / (1 + rs))
    
    for i in range(15, len(closes)):
        avg_gain = ((avg_gain * 13) + gains[i-1]) / 14
        avg_loss = ((avg_loss * 13) + losses[i-1]) / 14
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    bb_std = np.array([np.std(closes[max(0, i-20+1):i+1]) for i in range(len(closes))])
    bb_upper = sma20 + (bb_std * 2)
    bb_lower = sma20 - (bb_std * 2)
    
    # Create normalized features
    def normalize(x):
        std = np.std(x)
        return (x - np.mean(x)) / (std if std > 0 else 1)
    
    features = np.column_stack([
        normalize(closes),
        normalize(volumes),
        returns,  # Already normalized
        normalize(sma20 - closes),  # Distance from SMA20
        normalize(sma50 - closes),  # Distance from SMA50
        rsi / 100.0,  # Scale RSI to 0-1
        normalize(volumes - np.mean(volumes)),  # Volume trend
        normalize(np.diff(np.append(closes, closes[-1]))),  # Price momentum
        normalize(bb_upper - closes),  # Distance from upper BB
        normalize(closes - bb_lower),  # Distance from lower BB
        normalize(bb_std)  # Volatility
    ])
    
    return features

def fetch_stock_data(symbol, period='60d'):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if len(df) < 50:  # Need at least 50 days of data
            print(f"Insufficient data for {symbol} (need at least 50 days)")
            return None
        
        # Create numpy array with required columns
        data = []
        for date, row in df.iterrows():
            data.append([
                date.strftime('%Y-%m-%d'),
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Volume'],
                0.0 if len(data) == 0 else (row['Close'] - data[-1][4]) / data[-1][4]
            ])
        
        return np.array(data)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def predict_stocks(symbols, models_dir='trained_models'):
    """Make predictions for multiple stocks using unified model"""
    predictions = []
    
    # Load unified model
    model_path = os.path.join(models_dir, 'unified_model.joblib')
    if not os.path.exists(model_path):
        print("Unified model not found")
        return []
    
    model = joblib.load(model_path)
    
    for symbol in symbols:
        try:
            # Fetch recent data
            data = fetch_stock_data(symbol)
            if data is None:
                continue
                
            # Prepare features
            features = prepare_stock_features(data)
            if features is None or len(features) == 0:
                print(f"Could not prepare features for {symbol}")
                continue
            
            # Make prediction
            prob = model.predict_proba(features)[-1]  # Get prediction for most recent day
            stock = yf.Ticker(symbol)
            info = stock.info
            
            predictions.append({
                'symbol': symbol,
                'probability': prob[1],  # Probability of upward movement
                'current_price': float(data[-1][4]),  # Latest closing price
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', 0),
                'volume': float(data[-1][5]),  # Latest volume
                'sma20': float(np.mean(data[-20:, 4].astype(float))),  # 20-day moving average
                'rsi': float(features[-1, 5] * 100)  # Last RSI value
            })
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    # Sort by probability and get top 5
    predictions.sort(key=lambda x: x['probability'], reverse=True)
    return predictions[:5]

def main():
    # List of stocks to predict (same as training)
    symbols = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA'
    ]
    
    print("\nFetching current data and making predictions...")
    predictions = predict_stocks(symbols)
    
    print("\nTop 5 Stock Predictions:")
    print("=======================")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['company_name']} ({pred['symbol']})")
        print(f"   Sector: {pred['sector']}")
        print(f"   Current Price: ${pred['current_price']:,.2f}")
        print(f"   Market Cap: ${pred['market_cap']:,.0f}")
        print(f"   P/E Ratio: {pred['pe_ratio']:.2f}")
        print(f"   Daily Volume: {pred['volume']:,.0f}")
        print(f"   Upward Movement Probability: {pred['probability']*100:.1f}%")
        print("   " + "="*50)
        print()

if __name__ == "__main__":
    main()
