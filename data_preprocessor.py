import numpy as np
import os
import csv
from scipy import stats
from sklearn.preprocessing import RobustScaler

class StockDataPreprocessor:
    def __init__(self, data_dir='stock_data'):
        self.data_dir = data_dir
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
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
        """Handle missing or invalid values in the data"""
        numeric_data = data[:, 1:].astype(float)
        
        # Replace inf and -inf with NaN
        numeric_data = np.where(np.isinf(numeric_data), np.nan, numeric_data)
        
        # For each column, replace NaN with the mean of nearby values
        for col in range(numeric_data.shape[1]):
            mask = np.isnan(numeric_data[:, col])
            if np.any(mask):
                # Use rolling window mean for missing values
                window_size = 5
                for i in np.where(mask)[0]:
                    start = max(0, i - window_size)
                    end = min(len(numeric_data), i + window_size + 1)
                    valid_values = numeric_data[start:end, col][~np.isnan(numeric_data[start:end, col])]
                    if len(valid_values) > 0:
                        numeric_data[i, col] = np.mean(valid_values)
                    else:
                        # If no valid values in window, use column mean
                        valid_values = numeric_data[:, col][~np.isnan(numeric_data[:, col])]
                        numeric_data[i, col] = np.mean(valid_values)
        
        # Combine dates with processed numeric data
        return np.column_stack((data[:, 0], numeric_data))
    
    def remove_outliers(self, data, columns, threshold=3):
        """Remove extreme outliers using z-score method"""
        numeric_data = data[:, 1:].astype(float)
        
        # Calculate z-scores for each feature, handling NaN values
        z_scores = np.zeros_like(numeric_data)
        for col in range(numeric_data.shape[1]):
            valid_data = ~np.isnan(numeric_data[:, col])
            if np.sum(valid_data) > 0:
                col_mean = np.mean(numeric_data[valid_data, col])
                col_std = np.std(numeric_data[valid_data, col])
                if col_std > 0:
                    z_scores[valid_data, col] = np.abs((numeric_data[valid_data, col] - col_mean) / col_std)
        
        # Create a mask for non-outlier data points
        mask = (z_scores < threshold).all(axis=1)
        
        # Keep dates and apply mask to numeric data
        return np.column_stack((data[mask, 0], numeric_data[mask]))
    
    def scale_features(self, data, columns):
        """Scale features using RobustScaler"""
        numeric_data = data[:, 1:].astype(float)
        
        # Scale each feature
        scaled_data = self.scaler.fit_transform(numeric_data)
        
        # Combine dates with scaled numeric data
        return np.column_stack((data[:, 0], scaled_data))
    
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
        
        # 1. Price Range Percentage (with safe division)
        price_range = np.zeros_like(opens)
        valid_opens = opens != 0
        price_range[valid_opens] = (highs[valid_opens] - lows[valid_opens]) / opens[valid_opens] * 100
        
        # 2. Volume Weighted Average Price (VWAP)
        vwap = np.zeros_like(closes)
        valid_volumes = volumes != 0
        avg_prices = (highs + lows + closes) / 3
        vwap[valid_volumes] = (avg_prices[valid_volumes] * volumes[valid_volumes]) / volumes[valid_volumes]
        
        # 3. Price Momentum (rate of change)
        momentum = np.zeros_like(closes)
        valid_prev_closes = closes[:-1] != 0
        momentum[1:][valid_prev_closes] = (closes[1:][valid_prev_closes] - closes[:-1][valid_prev_closes]) / closes[:-1][valid_prev_closes] * 100
        
        # 4. Volume Force Index
        force_index = np.zeros_like(closes)
        force_index[1:] = (closes[1:] - closes[:-1]) * volumes[1:]
        
        # 5. True Range
        true_range = np.zeros_like(closes)
        prev_closes = np.roll(closes, 1)
        true_range[1:] = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - prev_closes[1:]),
                np.abs(lows[1:] - prev_closes[1:])
            )
        )
        
        # Add new features to the dataset
        new_features = np.column_stack((
            price_range,
            vwap,
            momentum,
            force_index,
            true_range
        ))
        
        # Combine all data
        combined_data = np.column_stack((data, new_features))
        
        # Add new column names
        new_columns = columns + [
            'PriceRange',
            'VWAP',
            'Momentum',
            'ForceIndex',
            'TrueRange'
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
            data = self.scale_features(data, columns)
            data, columns = self.add_derived_features(data, columns)
            
            # Save processed data
            output_filepath = os.path.join(self.data_dir, f'processed_{filename}')
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
