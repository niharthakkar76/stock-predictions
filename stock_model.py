import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import os
import glob
import csv
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        # Create base classifiers optimized for imbalanced data
        self.rf = RandomForestClassifier(
            n_estimators=200,  # Reduced from 2000
            max_depth=6,    # Reduced complexity
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',  # Better handling of imbalance
            bootstrap=True,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        
        self.xgb = XGBClassifier(
            n_estimators=200,  # Reduced from 2000
            max_depth=4,      # Reduced complexity
            learning_rate=0.1, # Increased for faster convergence
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=5,  # Adjusted for typical imbalance
            tree_method='hist',
            grow_policy='lossguide',
            n_jobs=-1,
            random_state=42
        )
        
        self.gb = GradientBoostingClassifier(
            n_estimators=200,  # Reduced from 1500
            max_depth=4,       # Reduced complexity
            learning_rate=0.1,  # Increased for faster convergence
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42
        )
        
        # Feature selector with better feature balance
        self.feature_selector = SelectFromModel(
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=12,
                colsample_bytree=0.8,  # Reduce feature dominance
                subsample=0.8,  # More robust feature selection
                random_state=42
            )
        )
        
        # Slightly increase minority samples
        self.sampler = SMOTE(
            sampling_strategy=0.35,  # Increase minority representation
            random_state=42,
            k_neighbors=5
        )
        
        # Simplified pipelines without redundant steps
        self.rf_pipeline = ImbPipeline([
            ('classifier', self.rf)
        ])
        
        self.xgb_pipeline = ImbPipeline([
            ('classifier', self.xgb)
        ])
        
        self.gb_pipeline = ImbPipeline([
            ('classifier', self.gb)
        ])
        
        # Ensemble with unanimous voting requirement
        self.model = VotingClassifier(
            estimators=[
                ('rf', self.rf_pipeline),
                ('xgb', self.xgb_pipeline),
                ('gb', self.gb_pipeline)
            ],
            voting='soft',
            weights=[1.5, 2, 1.5],  # Increased RF and GB weights
            n_jobs=-1
        )
        
        # Add threshold optimization
        self.threshold = 0.5  # Will be optimized during training
        
        self.data_dir = 'processed_data'
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
        
    def optimize_threshold(self, X, y):
        """Optimize threshold with strong precision focus"""
        # Get predicted probabilities from each classifier
        rf_proba = self.rf_pipeline.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_pipeline.predict_proba(X)[:, 1]
        gb_proba = self.gb_pipeline.predict_proba(X)[:, 1]
        
        # Require strong agreement between models
        y_scores = (rf_proba + xgb_proba + gb_proba) / 3
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
        
        # Find threshold that achieves at least 40% precision
        target_precision = 0.4
        valid_idx = precisions >= target_precision
        
        if np.any(valid_idx):
            # Among thresholds meeting precision target, choose one with best F0.5
            valid_precisions = precisions[valid_idx]
            valid_recalls = recalls[valid_idx]
            valid_thresholds = thresholds[valid_idx[:-1]]  # Account for length mismatch
            
            f_beta_scores = (1 + 0.25**2) * (valid_precisions * valid_recalls) / \
                           (0.25**2 * valid_precisions + valid_recalls)
            best_idx = np.argmax(f_beta_scores)
            self.threshold = valid_thresholds[best_idx]
        else:
            # If no threshold meets target, take highest precision threshold
            best_idx = np.argmax(precisions)
            self.threshold = thresholds[min(best_idx, len(thresholds)-1)]
        
        print(f"Selected threshold: {self.threshold:.3f}")
        print(f"Expected precision at threshold: {precisions[best_idx]:.3f}")
        print(f"Expected recall at threshold: {recalls[best_idx]:.3f}")
        
        return self.threshold
    
    def optimize_model_params(self, X, y):
        """Train model with SMOTE and optimize threshold"""
        print(f"Original class distribution: {Counter(y)}")
        
        # Fit the model (SMOTE is applied within each pipeline)
        self.model.fit(X, y)
        
        # Optimize classification threshold
        self.threshold = self.optimize_threshold(X, y)
        print(f"Optimized classification threshold: {self.threshold:.3f}")
        
        return self.model
        
    def load_data(self, filename):
        """Load preprocessed data from CSV file using pandas for efficiency"""
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return None, None
        
        try:
            # Read data efficiently with pandas
            df = pd.read_csv(filename)
            headers = df.columns.tolist()
            
            # Check for minimum required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in headers for col in required_cols):
                print(f"Missing required columns in {filename}")
                return None, None
            
            # Convert to numpy array efficiently
            data = df.to_numpy()
            
            # Check for sufficient valid data
            price_cols = data[:, 1:5].astype(float)
            valid_rows = ~np.isnan(price_cols).all(axis=1)
            if np.sum(valid_rows) < 50:
                print(f"Insufficient valid data in {filename}")
                return None, None
            
            return data[valid_rows], headers
            
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
        
        # Get all stock data files
        stock_files = glob.glob(os.path.join(self.data_dir, '*_data.csv'))
        if not stock_files:
            print("No stock data files found")
            return None
        
        # Process all stocks in parallel
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            results = list(executor.map(self.process_stock_data, stock_files))
        
        # Filter valid results
        valid_results = [(f, t) for f, t in results if f is not None and t is not None]
        if not valid_results:
            print("No valid data found for any stock")
            return None
        
        # Combine features and targets
        all_features, all_targets = zip(*valid_results)
        X = np.vstack(all_features)
        y = np.concatenate(all_targets)
        
        # Print data stats
        n_samples = len(y)
        n_positive = np.sum(y)
        print(f"\nTotal samples: {n_samples}")
        print(f"Positive samples: {n_positive} ({n_positive/n_samples:.2%})")
        
        # Check class distribution
        class_counts = np.bincount(y)
        print(f"\nInitial class distribution: {class_counts}")
        
        # Apply feature selection
        print("\nSelecting important features...")
        self.feature_selector.fit(X, y)
        importances = self.feature_selector.estimator_.feature_importances_
        threshold = np.percentile(importances, 50)  # Keep top 50% features
        feature_mask = importances >= threshold
        X = X[:, feature_mask]
        print(f"Selected {np.sum(feature_mask)} features")
        
        # Print feature importances
        sorted_idx = np.argsort(importances[feature_mask])
        print("\nTop feature importances:")
        for idx in sorted_idx[-5:]:
            print(f"Feature {idx}: {importances[feature_mask][idx]:.4f}")
        
        # Apply sampling with safeguards
        print("\nBalancing dataset...")
        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            # Verify we have both classes
            new_class_counts = np.bincount(y_resampled)
            if len(new_class_counts) < 2 or 0 in new_class_counts:
                print("Warning: Sampling produced invalid class distribution. Using original data.")
                X_resampled, y_resampled = X, y
            else:
                print(f"Resampled class distribution: {new_class_counts}")
            X, y = X_resampled, y_resampled
        except Exception as e:
            print(f"Warning: Sampling failed ({str(e)}). Using original data.")

        
        # Train-test split with time series consideration
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print("\nTraining models...")
        # Train each model separately with appropriate class handling
        for name, pipeline in [('RF', self.rf_pipeline), ('XGB', self.xgb_pipeline), ('GB', self.gb_pipeline)]:
            print(f"\nTraining {name}...")
            try:
                # Basic model configurations focused on handling imbalance
                if name == 'RF':
                    pipeline.named_steps['classifier'].set_params(
                        class_weight='balanced',
                        n_estimators=200,
                        max_depth=5,
                        min_samples_split=10,
                        min_samples_leaf=4
                    )
                elif name == 'XGB':
                    pipeline.named_steps['classifier'].set_params(
                        scale_pos_weight=12,
                        max_depth=4,
                        min_child_weight=1,
                        learning_rate=0.1
                    )
                elif name == 'GB':
                    # Improve GB performance with better params
                    pipeline.named_steps['classifier'].set_params(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=5,
                        subsample=0.8,
                        max_features='sqrt',
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                        learning_rate=0.05
                    )
                
                # Train model with basic configuration
                pipeline.fit(X_train, y_train)
                
                # Get probabilities
                y_pred_proba = pipeline.predict_proba(X_test)
                
                # Simple threshold optimization
                thresholds = np.arange(0.3, 0.7, 0.05)
                best_f1, best_threshold = 0, 0.5
                
                for threshold in thresholds:
                    y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
                    f1 = f1_score(y_test, y_pred)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                # Final prediction with best threshold
                y_pred = (y_pred_proba[:, 1] >= best_threshold).astype(int)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred)
                }
                print(f"Best threshold: {best_threshold:.2f}")
            
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
            
            print(f"{name} Performance:")
            for metric, score in metrics.items():
                print(f"{metric.capitalize()}: {score:.4f}")
        
        # Train final ensemble
        print("\nTraining final ensemble...")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        # Calculate final metrics
        final_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        print("\nEnsemble Performance:")
        for metric, score in final_metrics.items():
            print(f"{metric.capitalize()}: {score:.4f}")
        
        # Save models
        print("\nSaving models...")
        for name, pipeline in [('rf', self.rf_pipeline), ('xgb', self.xgb_pipeline), 
                             ('gb', self.gb_pipeline), ('ensemble', self.model)]:
            joblib.dump(pipeline, os.path.join(self.models_dir, f'{name}_model.joblib'))
        
        self.is_model_trained = True
        return final_metrics

    def process_stock_data(self, file_path):
        """Process a single stock's data in parallel"""
        try:
            features, target = self.prepare_features(file_path)
            if features is not None and target is not None:
                n_positive = np.sum(target)
                if len(target) >= 100 and n_positive >= 10:
                    return features, target
        except Exception as e:
            symbol = os.path.basename(file_path).replace('_data.csv', '')
            print(f"Error processing {symbol}: {str(e)}")
        return None, None

def main():
    predictor = StockPredictor()
    metrics = predictor.train_unified_model()
    
    if metrics:
        print("\nFinal Model Performance:")
        for metric, score in metrics.items():
            print(f"{metric.capitalize()}: {score:.4f}")

if __name__ == "__main__":
    main()
