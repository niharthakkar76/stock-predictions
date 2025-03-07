import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import os
import glob
import csv
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        # Create an ensemble of models optimized for stock prediction
        self.rf = RandomForestClassifier(
            n_estimators=1000,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced_subsample',
            bootstrap=True,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        
        self.xgb = XGBClassifier(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.005,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=10,
            scale_pos_weight=5,  # Further increased weight for minority class
            gamma=2,  # Increased regularization
            max_delta_step=1,  # Help with class imbalance
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1,  # L2 regularization
            n_jobs=-1,
            random_state=42
        )
        
        self.gb = GradientBoostingClassifier(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.005,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            validation_fraction=0.2,
            n_iter_no_change=50,
            random_state=42
        )
        
        # Ensemble model
        self.model = VotingClassifier(
            estimators=[
                ('rf', self.rf),
                ('xgb', self.xgb),
                ('gb', self.gb)
            ],
            voting='soft',
            n_jobs=-1
        )
        
        self.data_dir = 'stock_data'
        self.scaler = StandardScaler()
        self.models_dir = 'trained_models'
        self.is_model_trained = False
        
        # Create necessary directories
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # Create necessary directories
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
    def optimize_model_params(self, X, y):
        """Simple model optimization without GridSearchCV for faster training"""
        # Skip grid search and use pre-optimized parameters
        self.model.fit(X, y)
        return self.model
        
    def load_data(self, filename):
        """Load preprocessed data from CSV file"""
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return None, None
        
        try:
            # First pass: Read headers and count valid rows
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                # Count rows with valid price data
                valid_count = 0
                for row in reader:
                    if len(row) >= 5:  # At least Date, Open, High, Low, Close needed
                        if not all(val.lower() == 'nan' for val in row[1:5]):  # Check if all price fields are 'nan'
                            valid_count += 1
            
            if valid_count < 50:  # Skip if not enough valid data
                print(f"Insufficient valid data in {filename} (only {valid_count} valid rows)")
                return None, None
            
            # Second pass: Read the data
            data = []
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header since we already have it
                for row in reader:
                    # Keep the date as string, convert other fields
                    processed_row = [row[0]]  # Date stays as string
                    for val in row[1:]:
                        if val.lower() == 'nan':
                            processed_row.append(None)
                        else:
                            try:
                                processed_row.append(float(val))
                            except ValueError:
                                processed_row.append(None)
                    data.append(processed_row)
            
            return np.array(data, dtype=object), headers
            
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            return None

    def prepare_features(self, filename):
        """Prepare features using preprocessed data"""
        # Load preprocessed data
        result = self.load_data(filename)
        if result is None:
            return None, None
        
        data, headers = result
        if len(data) < 50:  # Need at least 50 data points
            print(f"Insufficient data points in {filename}")
            return None, None
        
        # Extract basic features
        try:
            # Get indices for basic columns
            basic_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            indices = {col: headers.index(col) for col in basic_cols}
            
            # Convert columns to float arrays, replacing None with np.nan
            opens = np.array([float(x) if x is not None else np.nan for x in data[:, indices['Open']]])
            highs = np.array([float(x) if x is not None else np.nan for x in data[:, indices['High']]])
            lows = np.array([float(x) if x is not None else np.nan for x in data[:, indices['Low']]])
            closes = np.array([float(x) if x is not None else np.nan for x in data[:, indices['Close']]])
            volumes = np.array([float(x) if x is not None else 0 for x in data[:, indices['Volume']]])
            
            # Remove rows where all price data is NaN
            valid_rows = ~(np.isnan(opens) & np.isnan(highs) & np.isnan(lows) & np.isnan(closes))
            if not np.any(valid_rows):
                print(f"No valid price data found in {filename}")
                return None, None
            
            # Filter data to keep only valid rows
            opens = opens[valid_rows]
            highs = highs[valid_rows]
            lows = lows[valid_rows]
            closes = closes[valid_rows]
            volumes = volumes[valid_rows]
            
            # Forward fill any remaining NaN values
            for arr in [opens, highs, lows, closes]:
                nan_indices = np.isnan(arr)
                last_valid_idx = np.where(~nan_indices)[0][0]
                arr[0:last_valid_idx] = arr[last_valid_idx]  # Fill leading NaNs
                for i in range(last_valid_idx + 1, len(arr)):
                    if nan_indices[i]:
                        arr[i] = arr[i-1]
            
        except (ValueError, IndexError) as e:
            print(f"Error extracting basic columns from {filename}: {str(e)}")
            return None, None
        
        # Calculate returns
        returns = np.zeros_like(closes)
        returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
        
        # Calculate technical indicators using numpy
        def calculate_sma(data, period):
            return np.array([np.mean(data[max(0, i-period+1):i+1]) for i in range(len(data))])
        
        def calculate_ema(data, period):
            alpha = 2 / (period + 1)
            ema = np.zeros_like(data)
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            return ema
        
        def calculate_rsi(prices, period=14):
            if len(prices) <= period:
                return np.zeros_like(prices)
            
            # Calculate price changes
            deltas = np.diff(prices)
            gains = np.zeros_like(deltas)
            losses = np.zeros_like(deltas)
            
            gains[deltas > 0] = deltas[deltas > 0]
            losses[deltas < 0] = -deltas[deltas < 0]
            
            # Calculate initial averages
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # Calculate RSI
            rsi = np.zeros_like(prices)
            if avg_loss == 0:
                rsi[period] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[period] = 100 - (100 / (1 + rs))
            
            # Calculate rest of RSI values
            for i in range(period + 1, len(prices)):
                avg_gain = ((avg_gain * (period - 1) + gains[i-1]) / period)
                avg_loss = ((avg_loss * (period - 1) + losses[i-1]) / period)
                
                if avg_loss == 0:
                    rsi[i] = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))
            
            return rsi
        
        # Calculate all technical indicators with proper padding
        def pad_indicator(indicator, lookback):
            # Pad the start with the first valid value
            if len(indicator) > lookback:
                indicator[:lookback] = indicator[lookback]
            return indicator
        
        # Moving averages
        sma20 = pad_indicator(calculate_sma(closes, 20), 20)
        sma50 = pad_indicator(calculate_sma(closes, 50), 50)
        
        # Bollinger Bands
        bb_middle = sma20
        rolling_std = np.array([np.std(closes[max(0, i-20+1):i+1]) for i in range(len(closes))])
        bb_upper = bb_middle + (rolling_std * 2)
        bb_lower = bb_middle - (rolling_std * 2)
        
        # MACD
        ema12 = pad_indicator(calculate_ema(closes, 12), 12)
        ema26 = pad_indicator(calculate_ema(closes, 26), 26)
        macd = ema12 - ema26
        signal_line = pad_indicator(calculate_ema(macd, 9), 9)
        
        # RSI
        rsi = calculate_rsi(closes)
        
        # VWAP
        typical_price = (highs + lows + closes) / 3
        vwap = np.zeros_like(closes)
        for i in range(len(closes)):
            start_idx = max(0, i-20+1)
            if i >= start_idx:
                vwap[i] = np.average(typical_price[start_idx:i+1], weights=volumes[start_idx:i+1])
            else:
                vwap[i] = typical_price[i]
        
        # Momentum (20-day)
        momentum = np.zeros_like(closes)
        if len(closes) > 20:
            momentum[20:] = closes[20:] - closes[:-20]
            momentum[:20] = momentum[20]  # Pad with first value
        
        # Force Index
        force_index = returns * volumes
        
        # Calculate future returns for different timeframes
        prediction_windows = [5, 10, 20]  # Multiple timeframes for prediction
        future_returns = np.zeros((len(closes), len(prediction_windows)))
        
        # Only calculate future returns if we have enough data
        min_required = max(prediction_windows)
        if len(closes) > min_required:
            for i, window in enumerate(prediction_windows):
                future_returns[:-window, i] = (closes[window:] - closes[:-window]) / closes[:-window]
            
            # Pad the end with zeros since we can't calculate future returns there
            future_returns[-min_required:] = 0
            
            # Technical signals
            price_above_sma = closes > sma20
            
            # Calculate momentum percentile excluding zeros
            nonzero_momentum = momentum[momentum != 0]
            if len(nonzero_momentum) > 0:
                momentum_threshold = np.percentile(nonzero_momentum, 75)
                strong_momentum = momentum > momentum_threshold
            else:
                strong_momentum = np.zeros_like(momentum, dtype=bool)
            
            oversold = rsi < 30
            
            # Calculate volume surge with proper handling of edge cases
            volume_means = np.array([np.mean(volumes[max(0, i-10):i+1]) for i in range(len(volumes))])
            volume_surge = volumes > (volume_means * 1.5)
            
            positive_force = force_index > 0
            
            # Combine technical signals
            technical_signal = ((price_above_sma & strong_momentum) | (oversold & volume_surge & positive_force))
            
            # Calculate target based on future returns and technical signals
            future_return_means = np.mean(future_returns, axis=1)
            if np.any(future_return_means != 0):
                threshold = np.percentile(future_return_means[future_return_means != 0], 70)
                future_return_signal = future_return_means > threshold
            else:
                future_return_signal = np.zeros_like(future_return_means, dtype=bool)
            
            target = (future_return_signal & technical_signal).astype(int)
            target[-min_required:] = 0  # No predictions for last N days to avoid lookahead bias
            
            # Create feature matrix only with valid data
            feature_matrix = np.column_stack([
                returns, sma20, sma50,
                bb_upper, bb_lower, bb_middle,
                macd, signal_line, rsi,
                momentum, force_index
            ])
            
            # Check for any remaining NaN or infinite values
            if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
                print(f"Found NaN or infinite values in features for {filename}")
                return None, None
            
            return feature_matrix, target
        else:
            print(f"Insufficient data for future returns calculation in {filename}")
            return None, None
        
        # Create normalized feature matrix
        def normalize_feature(data):
            std = np.std(data)
            if std == 0:
                return np.zeros_like(data)
            return (data - np.mean(data)) / std
        
        # Basic price and volume features
        basic_features = [
            normalize_feature(closes),
            normalize_feature(highs),
            normalize_feature(lows),
            normalize_feature(volumes),
            returns  # Already normalized
        ]
        
        # Technical indicators
        tech_features = [
            rsi / 100.0,  # Scale RSI to 0-1
            normalize_feature(momentum),
            normalize_feature(force_index),
            normalize_feature(sma20 - closes),  # Moving average crossovers
            normalize_feature(sma50 - closes),
            normalize_feature(bb_upper - closes),  # BB distances
            normalize_feature(bb_lower - closes),
            normalize_feature(macd),
            normalize_feature(signal_line)
        ]
        
        # Price patterns
        pattern_window = 5
        rolling_min = np.array([np.min(closes[max(0, i-pattern_window):i+1]) for i in range(len(closes))])
        rolling_max = np.array([np.max(closes[max(0, i-pattern_window):i+1]) for i in range(len(closes))])
        
        pattern_features = [
            (closes - rolling_min) / (rolling_max - rolling_min + 1e-8),  # Price position
            (closes <= rolling_min * 1.02).astype(float),  # Near support
            (closes >= rolling_max * 0.98).astype(float)   # Near resistance
        ]
        
        # Combine all features
        feature_matrix = np.column_stack(basic_features + tech_features + pattern_features)
        
        # Final validation
        if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
            print(f"Found invalid values in features for {filename}")
            return None, None
        raw_money_flow = typical_price * volumes
        
        mfi = np.zeros_like(closes)
        for i in range(window, len(closes)):
            pos_flow = np.sum(raw_money_flow[i-window:i][typical_price[i-window:i] > typical_price[i-window-1:i-1]])
            neg_flow = np.sum(raw_money_flow[i-window:i][typical_price[i-window:i] < typical_price[i-window-1:i-1]])
            mfi[i] = 100 - (100 / (1 + pos_flow / (neg_flow + 1e-10)))
        
        # On-Balance Volume (OBV)
        obv = np.zeros_like(closes)
        obv[1:] = np.where(closes[1:] > closes[:-1], volumes[1:],
                          np.where(closes[1:] < closes[:-1], -volumes[1:], 0)).cumsum()
        
        # Combine all features into a matrix
        feature_matrix = np.column_stack([
            closes, highs, lows, volumes, returns,
            rsi, momentum, force_index, sma20, sma50,
            bb_upper, bb_lower, bb_middle, macd, signal_line,
            vwap, near_support, near_resistance, trend_strength,
            mfi, obv/np.mean(volumes)
        ])
        
        # Remove any rows with NaN values
        valid_rows = ~np.isnan(feature_matrix).any(axis=1)
        feature_matrix = feature_matrix[valid_rows]
        target = target[valid_rows]
        
        # Scale features
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        return feature_matrix, target

    def train_model(self, symbol):
        """Train model for a specific stock with advanced features and optimization"""
        try:
            # Load and prepare features
            file_path = os.path.join(self.data_dir, f'{symbol}_data.csv')
            X, y = self.prepare_features(file_path)
            
            if len(X) < 100:  # Skip if not enough data
                return None
            
            # Split the data using time series split
            train_size = int(len(X) * 0.8)
            X_train = X[:train_size]
            X_test = X[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            # Optimize and train the model
            self.model = self.optimize_model_params(X_train, y_train)
            self.model.fit(X_train, y_train)
            
            # Save the trained model and scaler
            model_path = os.path.join(self.models_dir, f'{symbol}_model.joblib')
            scaler_path = os.path.join(self.models_dir, f'{symbol}_scaler.joblib')
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"Saved trained model for {symbol} to {model_path}")
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Load last data point for analysis
            dates, opens, highs, lows, closes, volumes, returns, sma20, sma50, \
            bb_upper, bb_lower, bb_middle, macd, signal_line, rsi = self.load_data(file_path)
            
            # Calculate signals
            price_trend = "Upward" if closes[-1] > closes[-5] else "Downward"
            rsi_signal = "Overbought" if rsi[-1] > 70 else "Oversold" if rsi[-1] < 30 else "Neutral"
            
            # Calculate trend strength using ADX if available
            # Calculate ADX using numpy
            def calculate_adx(high, low, close, period=14):
                # Calculate +DM and -DM
                high_diff = np.diff(high)
                low_diff = np.diff(low)
                
                pos_dm = np.where(
                    (high_diff > 0) & (high_diff > -low_diff),
                    high_diff,
                    0
                )
                neg_dm = np.where(
                    (low_diff < 0) & (-low_diff > high_diff),
                    -low_diff,
                    0
                )
                
                # Calculate TR
                high_low = high[1:] - low[1:]
                high_close = np.abs(high[1:] - close[:-1])
                low_close = np.abs(low[1:] - close[:-1])
                tr = np.maximum(high_low, np.maximum(high_close, low_close))
                
                # Smooth with EMA
                def smooth(x, period):
                    alpha = 1/period
                    smoothed = np.zeros_like(x)
                    smoothed[0] = x[0]
                    for i in range(1, len(x)):
                        smoothed[i] = alpha * x[i] + (1 - alpha) * smoothed[i-1]
                    return smoothed
                
                smoothed_tr = smooth(tr, period)
                smoothed_pos_dm = smooth(pos_dm, period)
                smoothed_neg_dm = smooth(neg_dm, period)
                
                # Calculate +DI and -DI
                pos_di = 100 * smoothed_pos_dm / smoothed_tr
                neg_di = 100 * smoothed_neg_dm / smoothed_tr
                
                # Calculate DX and ADX
                dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
                adx = smooth(dx, period)
                
                return adx
            
            try:
                adx = calculate_adx(highs, lows, closes)
                adx_strength = "Strong" if adx[-1] > 25 else "Weak"
            except:
                adx_strength = "Unknown"
            
            # Volume analysis
            volume_ma = np.mean(volumes[-20:])
            volume_signal = "High" if volumes[-1] > volume_ma else "Low"
            
            # Get feature names
            feature_names = [
                'Returns', 'Log_Returns', 'Volatility', 'Volume_Ratio',
                'Price_Momentum', 'Price_Acceleration', 'RSI', 'BB_Upper',
                'BB_Middle', 'BB_Lower'
            ]
            
            if X.shape[1] > len(feature_names):  # If we have advanced features
                feature_names.extend(['ADX', 'ADXR', 'MFI', 'CCI', 'ATR', 'OBV', 'AD', 'DOJI', 'HAMMER'])
            
            # Get feature importance
            feature_importance = dict(zip(feature_names, self.model.feature_importances_))
            
            return {
                'symbol': symbol,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'feature_importance': feature_importance,
                'last_close': closes[-1],
                'rsi': rsi[-1],
                'price_trend': price_trend,
                'rsi_signal': rsi_signal,
                'trend_strength': adx_strength,
                'volume_signal': volume_signal,
                'model_params': self.model.get_params()
            }
            
        except Exception as e:
            print(f"Error training model for {symbol}: {str(e)}")
            return None

    def arima_forecast(self, symbol, days=5):
        """Generate ARIMA forecast for a stock"""
        try:
            # Load data
            file_path = os.path.join(self.data_dir, f'{symbol}_data.csv')
            dates, opens, highs, lows, closes, *_ = self.load_data(file_path)
            
            # Fit ARIMA model
            model = ARIMA(closes, order=(5,1,0))
            results = model.fit()
            forecast = results.forecast(steps=days)
            
            return forecast.tolist()
        except Exception as e:
            print(f"Error generating ARIMA forecast for {symbol}: {str(e)}")
            return None

    def train_unified_model(self):
        """Train a unified model using preprocessed data from all stocks"""
        print("Training unified model...")
        
        # Get all stock data files (original files, not processed)
        stock_files = glob.glob(os.path.join(self.data_dir, '*_data.csv'))
        stock_files = [f for f in stock_files if not os.path.basename(f).startswith('processed_')]
        
        if not stock_files:
            print("No stock data files found")
            return None
        
        # Collect features and targets from all stocks
        all_features = []
        all_targets = []
        successful_stocks = []
        
        for file_path in stock_files:
            symbol = os.path.basename(file_path).replace('_data.csv', '')
            print(f"Processing {symbol}...")
            
            try:
                # Prepare features and target
                features, target = self.prepare_features(file_path)
                
                if features is not None and target is not None:
                    # Ensure we have enough samples and a balanced dataset
                    n_positive = np.sum(target)
                    if len(target) >= 100 and n_positive >= 10:  # At least 100 samples and 10 positive cases
                        all_features.append(features)
                        all_targets.append(target)
                        successful_stocks.append(symbol)
                        print(f"Successfully processed {symbol} data")
                        print(f"Shape: {features.shape}, Positive cases: {n_positive}")
                    else:
                        print(f"Insufficient or imbalanced data for {symbol}")
                else:
                    print(f"Failed to process {symbol} data")
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
        
        if not all_features:
            print("No valid data found for any stock")
            return None
        
        print(f"\nSuccessfully processed {len(successful_stocks)} stocks: {', '.join(successful_stocks)}")
        
        # Combine all data
        X = np.vstack(all_features)
        y = np.concatenate(all_targets)
        
        # Print class distribution
        n_samples = len(y)
        n_positive = np.sum(y)
        print(f"\nTotal samples: {n_samples}")
        print(f"Positive samples: {n_positive} ({n_positive/n_samples:.2%})")
        
        # Use TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        print("\nTraining with time series cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            fold_scores = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            print(f"\nFold {fold} scores:")
            for metric, score in fold_scores.items():
                cv_scores[metric].append(score)
                print(f"{metric.capitalize()}: {score:.4f}")
        
        # Print average metrics
        print("\nCross-validation Performance:")
        for metric, scores in cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"{metric.capitalize()}: {mean_score:.4f} (+/- {std_score:.4f})")
        
        # Final training on all data
        print("\nTraining final model on all data...")
        self.model.fit(X, y)
        
        # Save the model and scaler
        model_path = os.path.join(self.models_dir, 'unified_model.joblib')
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        
        self.is_model_trained = True
        
        # Return average metrics
        return {metric: np.mean(scores) for metric, scores in cv_scores.items()}

def main():
    predictor = StockPredictor()
    metrics = predictor.train_unified_model()
    
    if metrics:
        print("\nFinal Model Performance:")
        for metric, score in metrics.items():
            print(f"{metric.capitalize()}: {score:.4f}")

if __name__ == "__main__":
    main()
