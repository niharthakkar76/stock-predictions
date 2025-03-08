import numpy as np
import os
import csv
from scipy import stats
from sklearn.preprocessing import RobustScaler

class StockDataPreprocessor:
    def __init__(self, data_dir='stock_data', processed_dir='processed_data'):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        
        # Create processed data directory if it doesn't exist
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
    def load_stock_data(self, filename):
        """Load stock data from CSV and handle empty values"""
        data = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                # Handle empty values
                processed_row = []
                for i, x in enumerate(row):
                    if i == 0:  # Date column
                        processed_row.append(x)
                    else:
                        try:
                            # Replace empty strings with NaN
                            val = float(x) if x.strip() else np.nan
                            processed_row.append(val)
                        except ValueError:
                            processed_row.append(np.nan)
                data.append(processed_row)
        return np.array(data)
    
    def handle_missing_values(self, data, columns):
        """Handle missing or invalid values in the data with robust methods"""
        numeric_data = data[:, 1:].astype(float)
        
        # Replace inf, -inf, and zeros in price columns with NaN
        numeric_data = np.where(np.isinf(numeric_data), np.nan, numeric_data)
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in columns:
                idx = columns.index(col) - 1
                numeric_data[:, idx] = np.where(numeric_data[:, idx] == 0, np.nan, numeric_data[:, idx])
        
        # For each column, handle missing values with multiple methods
        for col in range(numeric_data.shape[1]):
            col_data = numeric_data[:, col]
            mask = np.isnan(col_data)
            
            if np.all(mask):
                # If all values are NaN, use a small positive number
                numeric_data[:, col] = 1e-6
                continue
            
            # First try rolling window median (more robust than mean)
            window_size = 5
            for i in np.where(mask)[0]:
                start = max(0, i - window_size)
                end = min(len(col_data), i + window_size + 1)
                valid_values = col_data[start:end][~np.isnan(col_data[start:end])]
                if len(valid_values) > 0:
                    col_data[i] = np.median(valid_values)
            
            # For any remaining NaNs, use forward fill
            mask = np.isnan(col_data)
            if np.any(mask):
                last_valid = np.nan
                for i in range(len(col_data)):
                    if not np.isnan(col_data[i]):
                        last_valid = col_data[i]
                    elif not np.isnan(last_valid):
                        col_data[i] = last_valid
            
            # For any remaining NaNs, use backward fill
            mask = np.isnan(col_data)
            if np.any(mask):
                last_valid = np.nan
                for i in range(len(col_data)-1, -1, -1):
                    if not np.isnan(col_data[i]):
                        last_valid = col_data[i]
                    elif not np.isnan(last_valid):
                        col_data[i] = last_valid
            
            # If still have NaNs, use column median or small positive number
            mask = np.isnan(col_data)
            if np.any(mask):
                valid_values = col_data[~np.isnan(col_data)]
                if len(valid_values) > 0:
                    col_data[mask] = np.median(valid_values)
                else:
                    col_data[mask] = 1e-6
            
            numeric_data[:, col] = col_data
        
        # Ensure no zeros in price columns
        for col in price_cols:
            if col in columns:
                idx = columns.index(col) - 1
                numeric_data[:, idx] = np.maximum(numeric_data[:, idx], 1e-6)
        
        return np.column_stack((data[:, 0], numeric_data))
    
    def remove_outliers(self, data, columns, threshold=3):
        """Remove extreme outliers using robust methods"""
        numeric_data = data[:, 1:].astype(float)
        dates = data[:, 0]
        
        # Process each column separately
        for col in range(numeric_data.shape[1]):
            col_data = numeric_data[:, col]
            
            # Skip if column is constant
            if np.all(col_data == col_data[0]):
                continue
            
            # Calculate quartiles and IQR
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) > 0:
                q1 = np.percentile(valid_data, 25)
                q3 = np.percentile(valid_data, 75)
                iqr = q3 - q1
                
                if iqr > 0:  # Only apply if we have variation
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    
                    # Clip outliers to bounds instead of removing them
                    numeric_data[:, col] = np.clip(col_data, lower_bound, upper_bound)
        
        # Ensure price columns remain positive
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in columns:
                idx = columns.index(col) - 1
                numeric_data[:, idx] = np.maximum(numeric_data[:, idx], 1e-6)
        
        return np.column_stack((dates, numeric_data))
    
    def scale_features(self, data, columns):
        """Scale features using min-max scaling with safety checks"""
        dates = data[:, 0]
        numeric_data = data[:, 1:].astype(float)
        scaled_data = np.zeros_like(numeric_data)
        
        # Scale each column independently
        for col in range(numeric_data.shape[1]):
            col_data = numeric_data[:, col]
            
            # Skip scaling if column is constant
            if np.all(col_data == col_data[0]):
                scaled_data[:, col] = 0  # Center constant columns at 0
                continue
            
            # Get valid data for scaling
            valid_data = col_data[~np.isnan(col_data) & ~np.isinf(col_data)]
            if len(valid_data) > 0:
                # Use percentiles instead of min/max for robustness
                col_min = np.percentile(valid_data, 1)
                col_max = np.percentile(valid_data, 99)
                
                # Ensure we have a valid range
                if col_max > col_min:
                    # Scale to [-1, 1] range
                    scaled_data[:, col] = 2 * (col_data - col_min) / (col_max - col_min) - 1
                else:
                    scaled_data[:, col] = 0
            else:
                scaled_data[:, col] = 0
            
            # Clip outliers
            scaled_data[:, col] = np.clip(scaled_data[:, col], -3, 3)
        
        # Handle any remaining NaN or inf values
        scaled_data = np.nan_to_num(scaled_data, nan=0, posinf=3, neginf=-3)
        
        return np.column_stack((dates, scaled_data))
    
    def add_derived_features(self, data, columns):
        """Add derived features to enhance the dataset"""
        dates = data[:, 0]
        numeric_data = data[:, 1:].astype(float)
        
        # Extract basic price and volume data
        opens = numeric_data[:, columns.index('Open') - 1]
        highs = numeric_data[:, columns.index('High') - 1]
        lows = numeric_data[:, columns.index('Low') - 1]
        closes = numeric_data[:, columns.index('Close') - 1]
        volumes = numeric_data[:, columns.index('Volume') - 1]
        
        # Ensure price data consistency
        opens = np.maximum(opens, 1e-6)
        highs = np.maximum(highs, opens)
        lows = np.minimum(lows, opens)
        lows = np.maximum(lows, 1e-6)
        closes = np.maximum(closes, 1e-6)
        
        # Handle volume data - ensure we never have zero volumes
        volumes = np.maximum(volumes, 0)
        # Calculate minimum volume for safety
        positive_volumes = volumes[volumes > 0]
        if len(positive_volumes) > 0:
            min_volume = np.percentile(positive_volumes, 1)  # Use 1st percentile as minimum
            if min_volume == 0:
                min_volume = np.mean(positive_volumes) * 0.01
        else:
            min_volume = 1.0
            
        # Add small noise to zero volumes to prevent normalization issues
        safe_volumes = np.where(volumes > 0, 
                               volumes, 
                               min_volume * (0.9 + 0.2 * np.random.random(volumes.shape)))
        
        def safe_divide(a, b, fill_value=0):
            """Safe division that handles zeros and NaNs"""
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.where(b != 0, a / b, fill_value)
                return np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=-fill_value)
        
        def calculate_sma(data, period):
            """Calculate Simple Moving Average with safety checks"""
            sma = np.zeros_like(data)
            for i in range(len(data)):
                start_idx = max(0, i - period + 1)
                window = data[start_idx:i+1]
                sma[i] = np.mean(window)
            return sma
        
        def calculate_ema(data, period):
            """Calculate EMA with safety checks"""
            # Start with SMA for more stability
            ema = calculate_sma(data, period)
            alpha = 2 / (period + 1)
            
            # Then calculate EMA
            for i in range(period, len(data)):
                ema[i] = data[i] * alpha + ema[i-1] * (1 - alpha)
            
            # Handle edge cases
            ema = np.nan_to_num(ema, nan=data[0], posinf=data[0], neginf=data[0])
            return ema
        
        # 1. Enhanced Price Action Features with bounds
        price_range = np.zeros_like(opens)
        price_range = np.clip((highs - lows) / opens * 100, -100, 100)
        
        # 2. Advanced Volume Analysis with safety checks
        typical_price = (highs + lows + closes) / 3
        vwap = np.zeros_like(closes)
        for i in range(len(vwap)):
            start_idx = max(0, i-20)
            vol_window = volumes[start_idx:i+1]
            price_window = typical_price[start_idx:i+1]
            vwap[i] = np.average(price_window, weights=vol_window)
        
        # Calculate volume EMAs
        volume_ema10 = calculate_ema(safe_volumes, 10)
        
        # Calculate volume indicators with safety checks
        volume_sma10 = calculate_sma(safe_volumes, 10)
        volume_ratio = safe_divide(safe_volumes, volume_sma10, fill_value=1.0)
        volume_ratio = np.clip(volume_ratio, 0.1, 10)  # Limit extreme ratios
        
        # 3. Momentum and Trend Indicators with bounds
        momentum_1d = np.zeros_like(closes)
        momentum_5d = np.zeros_like(closes)
        momentum_10d = np.zeros_like(closes)
        
        for i in range(1, len(closes)):
            momentum_1d[i] = np.clip((closes[i] - closes[i-1]) / closes[i-1] * 100, -50, 50)
            if i >= 5:
                momentum_5d[i] = np.clip((closes[i] - closes[i-5]) / closes[i-5] * 100, -100, 100)
            if i >= 10:
                momentum_10d[i] = np.clip((closes[i] - closes[i-10]) / closes[i-10] * 100, -150, 150)
        
        # Volatility Features
        def calculate_true_range():
            tr = np.zeros_like(closes)
            prev_closes = np.roll(closes, 1)
            prev_closes[0] = closes[0]
            
            for i in range(1, len(closes)):
                high_low = highs[i] - lows[i]
                high_pc = abs(highs[i] - closes[i-1])
                low_pc = abs(lows[i] - closes[i-1])
                tr[i] = max(high_low, high_pc, low_pc)
            
            return np.clip(tr, 0, np.inf)
        
        true_range = calculate_true_range()
        atr14 = calculate_ema(true_range, 14)
        
        # Price Pattern Features
        def safe_price_patterns():
            upper = np.maximum(opens, closes)
            lower = np.minimum(opens, closes)
            upper_shadow = np.maximum(highs - upper, 0)
            lower_shadow = np.maximum(lower - lows, 0)
            body = np.abs(closes - opens)
            return upper_shadow, lower_shadow, body
        
        upper_shadow, lower_shadow, body_size = safe_price_patterns()
        
        # Market Regime Indicators
        ema20 = calculate_ema(closes, 20)
        ema50 = calculate_ema(closes, 50)
        
        # Trend strength
        trend_strength = np.zeros_like(closes)
        trend_strength[20:] = safe_divide(ema20[20:] - ema50[20:], ema50[20:], fill_value=0) * 100
        trend_strength = np.clip(trend_strength, -100, 100)
        
        # Volatility regime
        volatility_regime = np.zeros_like(closes)
        volatility_regime[14:] = safe_divide(atr14[14:], closes[14:], fill_value=0) * 100
        volatility_regime = np.clip(volatility_regime, 0, 100)
        
        # Calculate VWAP using exponential weighted moving average for stability
        typical_price = (highs + lows + closes) / 3
        vwap = np.copy(typical_price)  # Start with typical price as base
        alpha = 2.0 / (20 + 1)  # Decay factor for 20-day window
        
        # Initialize with first valid value
        vwap[0] = typical_price[0]
        
        # Calculate VWAP using exponential weighting
        for i in range(1, len(closes)):
            # Weight is combination of volume and exponential decay
            weight = safe_volumes[i] * (1 - alpha)
            vwap[i] = (typical_price[i] * alpha + vwap[i-1] * weight) / (alpha + weight)
        
        # Ensure VWAP stays within reasonable bounds
        vwap = np.clip(vwap, np.minimum.reduce([opens, highs, lows, closes]), 
                             np.maximum.reduce([opens, highs, lows, closes]))
        
        # Calculate momentum indicators with safety
        def calculate_momentum(period):
            momentum = np.zeros_like(closes)
            for i in range(period, len(closes)):
                momentum[i] = safe_divide(closes[i] - closes[i-period], closes[i-period], fill_value=0) * 100
            return np.clip(momentum, -100, 100)
        
        momentum_1d = calculate_momentum(1)
        momentum_5d = calculate_momentum(5)
        momentum_10d = calculate_momentum(10)
        
        # Add new features to the dataset
        new_features = np.column_stack((
            price_range,          # Price action
            vwap,                 # Volume-weighted price
            volume_ratio,         # Volume trend
            momentum_1d,          # Short-term momentum
            momentum_5d,          # Medium-term momentum
            momentum_10d,         # Long-term momentum
            true_range,          # Volatility
            atr14,               # Average True Range
            upper_shadow,         # Price patterns
            lower_shadow,
            body_size,
            trend_strength,       # Market regime
            volatility_regime
        ))
        
        # Final safety check: replace any remaining NaN/inf values
        new_features = np.nan_to_num(new_features, nan=0, posinf=0, neginf=0)
        
        # Combine all data
        combined_data = np.column_stack((data, new_features))
        
        # Add new column names
        new_columns = columns + [
            'PriceRange',
            'VWAP',
            'VolumeRatio',
            'Momentum1D',
            'Momentum5D',
            'Momentum10D',
            'TrueRange',
            'ATR14',
            'UpperShadow',
            'LowerShadow',
            'BodySize',
            'TrendStrength',
            'VolatilityRegime'
        ]
        
        return combined_data, new_columns
    
    def process_file(self, filename):
        """Process a single stock data file"""
        try:
            # Load data
            filepath = os.path.join(self.data_dir, filename)
            data = self.load_stock_data(filepath)
            
            # Get column names from first row
            with open(filepath, 'r') as f:
                columns = next(csv.reader(f))
            
            # Process data
            data = self.handle_missing_values(data, columns)
            data = self.remove_outliers(data, columns)
            data, columns = self.add_derived_features(data, columns)
            data = self.scale_features(data, columns)  # Scale after adding features
            
            # Save processed data in the processed directory
            output_filepath = os.path.join(self.processed_dir, filename)
            with open(output_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(data)
            
            print(f"Successfully processed {filename}")
            return True
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return False
    
    def process_all_files(self):
        """Process all stock data files in the directory"""
        success_count = 0
        files = [f for f in os.listdir(self.data_dir) if f.endswith('_data.csv') and not f.startswith('processed_')]
        
        for filename in files:
            if self.process_file(filename):
                success_count += 1
        
        print(f"\nSuccessfully processed {success_count} out of {len(files)} files")

if __name__ == "__main__":
    preprocessor = StockDataPreprocessor()
    preprocessor.process_all_files()
