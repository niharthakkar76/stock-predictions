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

# Configure numpy random state
np.random.seed(42)
rng = np.random.RandomState(42)

class StockPredictor:
    def __init__(self):
        # Initialize feature columns
        self.feature_cols = [
            # Technical features
            'Returns', 'SMA20', 'SMA50', 'BB_Upper', 'BB_Lower',
            'BB_Middle', 'MACD', 'Signal_Line', 'RSI', 'Force_Index',
            'Volume_Ratio',
            # Fundamental features
            'PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'Market_Cap',
            'Debt_To_Equity', 'ROE', 'ROA', 'Quick_Ratio', 'Beta',
            'EPS_Growth'
        ]
        
        # Initialize base classifiers with scikit-learn models
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight='balanced',
            random_state=42
        )
        
        self.gb1 = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        self.gb2 = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=43
        )
        
        # Create voting classifier with weighted contributions
        self.model = VotingClassifier(
            estimators=[
                ('rf', self.rf),
                ('gb1', self.gb1),
                ('gb2', self.gb2)
            ],
            voting='soft',
            weights=[1.5, 2.5, 1.5]
        )
        
        # SMOTE sampler for handling imbalanced data
        self.sampler = SMOTE(
            sampling_strategy=0.35,
            random_state=42,
            k_neighbors=5
        )
        
        # Feature selector
        self.feature_selector = SelectFromModel(
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=12,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        )
        
        # SMOTE sampler
        self.sampler = SMOTE(
            sampling_strategy=0.35,
            random_state=42,
            k_neighbors=5
        )
        
        # Set up directories
        self.data_dir = 'processed_data'
        self.models_dir = 'trained_models'
        
        # Create necessary directories
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return np.zeros_like(prices)
        
        sma = np.zeros_like(prices)
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
        sma[:period-1] = sma[period-1]  # Pad start with first valid value
        return sma
    
    def calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.zeros_like(prices)
        
        ema = np.zeros_like(prices)
        multiplier = 2 / (period + 1)
        
        # Initialize EMA with SMA
        ema[period-1] = np.mean(prices[:period])
        
        # Calculate EMA
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        # Pad start with first valid value
        ema[:period-1] = ema[period-1]
        return ema
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return np.zeros_like(prices)
        
        # Calculate price changes
        deltas = np.zeros_like(prices)
        deltas[1:] = prices[1:] - prices[:-1]
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Initialize averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Calculate RSI
        rsi = np.zeros_like(prices)
        
        # First RSI value
        if avg_loss == 0:
            rsi[period-1] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[period-1] = 100 - (100 / (1 + rs))
        
        # Calculate remaining RSI values
        for i in range(period, len(prices)):
            avg_gain = ((avg_gain * (period - 1) + gains[i-1]) / period)
            avg_loss = ((avg_loss * (period - 1) + losses[i-1]) / period)
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        # Pad start with first valid value
        rsi[:period-1] = rsi[period-1]
        return rsi
        
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
        
        # Feature importance analysis
        print("\nAnalyzing feature importance...")
        self.xgb.fit(X, y)
        feature_cols = [
            # Technical features
            'Returns', 'SMA20', 'SMA50', 'BB_Upper', 'BB_Lower',
            'BB_Middle', 'MACD', 'Signal_Line', 'RSI',
            # Fundamental features
            'PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'Market_Cap',
            'Debt_To_Equity', 'ROE', 'ROA', 'Quick_Ratio', 'Beta',
            'EPS_Growth'
        ]
        importances = sorted(zip(feature_cols, self.xgb.feature_importances_),
                           key=lambda x: x[1], reverse=True)
        print("\nTop 10 most important features:")
        for feat, imp in importances[:10]:
            print(f"{feat}: {imp:.4f}")
        
        # Fit the model (SMOTE is applied within each pipeline)
        self.model.fit(X, y)
        
        # Optimize classification threshold
        self.threshold = self.optimize_threshold(X, y)
        print(f"\nOptimized classification threshold: {self.threshold:.3f}")
        
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
            
            # Technical features
            technical_features = [
                'Returns', 'SMA20', 'SMA50', 'BB_Upper', 'BB_Lower',
                'BB_Middle', 'MACD', 'Signal_Line', 'RSI'
            ]
            
            # Fundamental features
            fundamental_features = [
                'PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'Market_Cap',
                'Debt_To_Equity', 'ROE', 'ROA', 'Quick_Ratio', 'Beta',
                'EPS_Growth'
            ]
            
            # Combine all features
            feature_columns = technical_features + fundamental_features
            
            # Handle missing values in fundamental data
            for col in fundamental_features:
                if col in df.columns:
                    # Fill missing values with forward fill first, then backward fill
                    df[col] = df[col].fillna(method='ffill')
                    df[col] = df[col].fillna(method='bfill')
                    # If still missing, use median
                    df[col] = df[col].fillna(df[col].median())
                    # Handle infinite values with median of non-infinite values
                    mask = np.isinf(df[col])
                    if mask.any():
                        median_val = df.loc[~mask, col].median()
                        df.loc[mask, col] = median_val
                else:
                    print(f"Warning: Missing fundamental feature {col}")
                    # For missing columns, try to derive from other columns
                    if col == 'PE_Ratio' and 'Market_Cap' in df.columns and 'EPS_Growth' in df.columns:
                        df[col] = df['Market_Cap'] / (df['EPS_Growth'] + 1e-10)
                    elif col == 'ROE' and 'Net_Income' in df.columns and 'Total_Equity' in df.columns:
                        df[col] = df['Net_Income'] / (df['Total_Equity'] + 1e-10)
                    else:
                        df[col] = 0  # Use 0 as last resort
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

    def prepare_features(self, filename, df=None):
        """Prepare features using preprocessed data"""
        try:
            # Load preprocessed data if not provided
            if df is None:
                df = pd.read_csv(filename)
                if df.empty:
                    print(f"Empty dataframe in {filename}")
                    return None, None
            
            # Ensure required columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Missing required columns in {filename}")
                return None, None
            
            # Extract basic data
            data = df[required_cols].values
            dates = pd.to_datetime(data[:, 0])
            opens = data[:, 1].astype(float)
            highs = data[:, 2].astype(float)
            lows = data[:, 3].astype(float)
            closes = data[:, 4].astype(float)
            volumes = data[:, 5].astype(float)
            
            # Handle missing or invalid data
            valid_mask = ~(np.isnan(opens) | np.isnan(highs) | 
                         np.isnan(lows) | np.isnan(closes) | 
                         np.isnan(volumes))
            if np.sum(valid_mask) < 50:
                print(f"Insufficient valid data in {filename}")
                return None, None
            
            # Apply mask to all data
            dates = dates[valid_mask]
            opens = opens[valid_mask]
            highs = highs[valid_mask]
            lows = lows[valid_mask]
            closes = closes[valid_mask]
            volumes = volumes[valid_mask]
            
            # Calculate returns and technical indicators
            returns = np.zeros_like(closes)
            returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
            
            # Calculate technical indicators with error handling
            try:
                sma20 = self.calculate_sma(closes, 20)
                sma50 = self.calculate_sma(closes, 50)
                bb_middle = sma20
                rolling_std = np.array([np.std(closes[max(0, i-20+1):i+1]) for i in range(len(closes))])
                bb_upper = bb_middle + (rolling_std * 2)
                bb_lower = bb_middle - (rolling_std * 2)
                
                # MACD
                ema12 = self.calculate_ema(closes, 12)
                ema26 = self.calculate_ema(closes, 26)
                macd = ema12 - ema26
                signal_line = self.calculate_ema(macd, 9)
                
                # RSI and Force Index
                rsi = self.calculate_rsi(closes)
                force_index = returns * volumes
                
                # Create feature dictionary
                feature_dict = {
                    # Technical features
                    'Returns': returns,
                    'SMA20': sma20,
                    'SMA50': sma50,
                    'BB_Upper': bb_upper,
                    'BB_Lower': bb_lower,
                    'BB_Middle': bb_middle,
                    'MACD': macd,
                    'Signal_Line': signal_line,
                    'RSI': rsi,
                    'Force_Index': force_index,
                    'Volume_Ratio': volumes / np.mean(volumes)
                }
            except Exception as e:
                print(f"Error calculating technical indicators: {str(e)}")
                return None, None
            
            # Add fundamental features if available
            fundamental_features = ['PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'Market_Cap',
                                  'Debt_To_Equity', 'ROE', 'ROA', 'Quick_Ratio', 'Beta',
                                  'EPS_Growth']
            
            # Check if we have any fundamental features
            has_fundamentals = any(col in df.columns for col in fundamental_features)
            if not has_fundamentals:
                print(f"Warning: No fundamental features found in {filename}")
            
            for col in fundamental_features:
                if col in df.columns:
                    # Handle missing values in fundamental data
                    values = df[col].values[valid_mask]
                    values = np.nan_to_num(values, nan=0.0)
                    feature_dict[col] = values
                else:
                    feature_dict[col] = np.zeros_like(closes)
            
            # Create feature matrix in the same order as self.feature_cols
            try:
                feature_matrix = np.column_stack([feature_dict[col] for col in self.feature_cols])
            except KeyError as e:
                print(f"Missing feature column: {str(e)}")
                return None, None
            except Exception as e:
                print(f"Error creating feature matrix: {str(e)}")
                return None, None
            
            # Calculate target based on future returns
            target = np.zeros_like(closes, dtype=int)
            future_returns = np.zeros_like(closes)
            for i in range(len(closes)-5):
                future_returns[i] = (closes[i+5] - closes[i]) / closes[i]
            target[:-5] = (future_returns[:-5] > np.percentile(future_returns[:-5], 70)).astype(int)
            
            return feature_matrix, target
            
        except Exception as e:
            print(f"Error preparing features for {filename}: {str(e)}")
            return None, None

        


        def calculate_technical_features(closes, highs, lows, volumes):
            """Calculate technical indicators following exchange standards"""
            # Initialize basic features
            returns = np.zeros_like(closes)
            returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
            
            # RSI (14-period)
            rsi = self.calculate_rsi(closes)
            
            # Moving averages and Bollinger Bands
            sma20 = self.calculate_sma(closes, 20)
            sma50 = self.calculate_sma(closes, 50)
            rolling_std = np.array([np.std(closes[max(0, i-20+1):i+1]) for i in range(len(closes))])
            bb_upper = sma20 + (rolling_std * 2)
            bb_lower = sma20 - (rolling_std * 2)
            
            # MACD (12/26/9)
            ema12 = self.calculate_ema(closes, 12)
            ema26 = self.calculate_ema(closes, 26)
            macd = ema12 - ema26
            signal_line = self.calculate_ema(macd, 9)
            
            # Force Index and Volume Ratio
            force_index = returns * volumes
            volume_ratio = volumes / np.mean(volumes)
            
            return {
                'Returns': returns,
                'SMA20': sma20,
                'SMA50': sma50,
                'BB_Upper': bb_upper,
                'BB_Lower': bb_lower,
                'BB_Middle': sma20,
                'MACD': macd,
                'Signal_Line': signal_line,
                'RSI': rsi,
                'Force_Index': force_index,
                'Volume_Ratio': volume_ratio
            }
        
        def get_fundamental_features(df, valid_mask, exchange):
            """Extract fundamental features with exchange-specific handling"""
            fundamental_features = {
                'PE_Ratio': {'clip': (0, 200)},           # Cap extreme ratios
                'PB_Ratio': {'clip': (0, 50)},            # Cap extreme ratios
                'Dividend_Yield': {'clip': (0, 0.25)},    # Cap at 25%
                'Market_Cap': {'log': True},              # Log transform
                'Debt_To_Equity': {'clip': (0, 10)},      # Cap extreme ratios
                'ROE': {'clip': (-1, 1)},                # Normalize to [-100%, 100%]
                'ROA': {'clip': (-1, 1)},                # Normalize to [-100%, 100%]
                'Quick_Ratio': {'clip': (0, 5)},         # Cap extreme ratios
                'Beta': {'clip': (-2, 4)},               # Typical beta range
                'EPS_Growth': {'clip': (-1, 2)}          # Cap extreme growth
            }
            
            feature_dict = {}
            for col, params in fundamental_features.items():
                if col in df.columns:
                    values = df[col].values[valid_mask]
                    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    if params.get('log', False):
                        # Log transform with sign preservation
                        signs = np.sign(values)
                        values = signs * np.log1p(np.abs(values))
                    
                    if 'clip' in params:
                        values = np.clip(values, *params['clip'])
                    
                    feature_dict[col] = values
                else:
                    feature_dict[col] = np.zeros(len(valid_mask))
            
            return feature_dict
        
        # Extract exchange from filename
        exchange = os.path.basename(filename).split('_')[0] if '_' in os.path.basename(filename) else 'NYSE'
        
        # Calculate technical features
        tech_features = calculate_technical_features(closes, highs, lows, volumes)
        
        # Get fundamental features with exchange-specific handling
        fund_features = get_fundamental_features(df, valid_mask, exchange)
        
        # Combine all features
        feature_dict = {**tech_features, **fund_features}
        
        # Create feature matrix in the same order as self.feature_cols
        try:
            feature_matrix = np.column_stack([feature_dict[col] for col in self.feature_cols])
            
            # Calculate target with exchange-specific thresholds
            threshold_map = {
                'NYSE': 0.70,    # More liquid, use higher threshold
                'NASDAQ': 0.70,
                'DOW': 0.70,
                'FTSE': 0.65,    # Less liquid, use lower threshold
                'DAX': 0.65
            }
            threshold = threshold_map.get(exchange, 0.70)
            
            # Calculate future returns
            target = np.zeros_like(closes, dtype=int)
            future_returns = np.zeros_like(closes)
            for i in range(len(closes)-5):
                future_returns[i] = (closes[i+5] - closes[i]) / closes[i]
            
            # Set target based on exchange-specific threshold
            target[:-5] = (future_returns[:-5] > np.percentile(future_returns[:-5], threshold * 100)).astype(int)
            
            # Final validation
            if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
                print(f"Found NaN or infinite values in features for {filename}")
                return None, None
            
            return feature_matrix, target
            
        except Exception as e:
            print(f"Error creating feature matrix for {filename}: {str(e)}")
            return None, None
        
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
            # Configure numpy random state before saving
            np.random.set_state(rng.get_state())
            joblib.dump(self.model, model_path)
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
        
        # Apply feature selection while preserving fundamental features
        print("\nSelecting important features...")
        self.feature_selector.fit(X, y)
        importances = self.feature_selector.estimator_.feature_importances_
        
        # Always keep fundamental features
        fundamental_indices = [i for i, col in enumerate(self.feature_cols) 
                             if col in ['PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'Market_Cap',
                                       'Debt_To_Equity', 'ROE', 'ROA', 'Quick_Ratio', 'Beta',
                                       'EPS_Growth']]
        
        # For technical features, keep top 50%
        technical_indices = [i for i, col in enumerate(self.feature_cols) 
                           if i not in fundamental_indices]
        technical_importances = importances[technical_indices]
        threshold = np.percentile(technical_importances, 50)
        
        # Create feature mask
        feature_mask = np.zeros_like(importances, dtype=bool)
        feature_mask[fundamental_indices] = True  # Always keep fundamental features
        feature_mask[technical_indices] = importances[technical_indices] >= threshold
        
        X = X[:, feature_mask]
        selected_features = [col for i, col in enumerate(self.feature_cols) if feature_mask[i]]
        print(f"Selected {np.sum(feature_mask)} features:")
        print("Fundamental features:", [f for f in selected_features if f in ['PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'Market_Cap',
                                                                                'Debt_To_Equity', 'ROE', 'ROA', 'Quick_Ratio', 'Beta',
                                                                                'EPS_Growth']])
        print("Technical features:", [f for f in selected_features if f not in ['PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'Market_Cap',
                                                                                'Debt_To_Equity', 'ROE', 'ROA', 'Quick_Ratio', 'Beta',
                                                                                'EPS_Growth']])
        
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
        # Train the ensemble model
        print("\nTraining ensemble model...")
        self.model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        print("\nModel Performance:")
        for metric, score in metrics.items():
            print(f"{metric.capitalize()}: {score:.4f}")
        
        # Save the model
        print("\nSaving model...")
        model_path = os.path.join(self.models_dir, 'ensemble_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f, protocol=4)
        print(f"Saved model to {model_path}")
        
        self.is_model_trained = True
        return metrics

    def process_stock_data(self, file_path):
        """Process a single stock's data in parallel"""
        try:
            # Extract symbol and exchange from file path
            filename = os.path.basename(file_path)
            if '_' in filename:
                exchange, symbol = filename.split('_')[0:2]
                symbol = symbol.replace('_data.csv', '')
            else:
                symbol = filename.replace('_data.csv', '')
                exchange = 'NYSE'  # Default to NYSE if no exchange prefix
            
            # Load data with pandas
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"Error processing {symbol}: Empty data")
                return None, None
            
            # Add exchange-specific suffix for fundamental data
            if exchange == 'FTSE':
                symbol = f"{symbol}.L"
            elif exchange == 'DAX':
                symbol = f"{symbol}.DE"
            
            # Prepare features
            features, target = self.prepare_features(file_path, df)
            if features is not None and target is not None:
                n_positive = np.sum(target)
                if len(target) >= 100 and n_positive >= 10:
                    return features, target
                
        except Exception as e:
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
