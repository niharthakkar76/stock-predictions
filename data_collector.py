import numpy as np
import csv
import os
from datetime import datetime, timedelta
import yfinance as yf

class StockDataCollector:
    def __init__(self):
        self.data_dir = 'stock_data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Additional 50 diverse companies not in our existing dataset
        self.symbols = [
            # Healthcare & Biotech
            'UNH', 'PFE', 'NVO', 'AZN', 'NVS', 'GILD', 'REGN', 'VRTX', 'BIIB',
            # Finance & Fintech
            'BRK-B', 'BAC', 'C', 'GS', 'BLK', 'SCHW', 'AXP', 'SPGI', 'COIN',
            # Consumer & Retail
            'NKE', 'SBUX', 'TGT', 'HD', 'LOW', 'LULU', 'EL', 'ULTA', 'DG',
            # Industrial & Manufacturing
            'GE', 'MMM', 'BA', 'LMT', 'NOC', 'ROP', 'ETN', 'EMR',
            # Energy & Clean Tech
            'NEE', 'ENPH', 'SEDG', 'BE', 'PLUG',
            # Real Estate & Infrastructure
            'AMT', 'PLD', 'CCI', 'DLR',
            # Materials & Chemicals
            'LIN', 'APD', 'ECL', 'DD',
            # Transportation & Logistics
            'UPS', 'FDX', 'UAL', 'DAL'
        ]

    def download_stock_data(self, symbol, period='2y'):
        """Download stock data using yfinance and save as numpy arrays in CSV"""
        try:
            # Download data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if len(df) < 100:  # Skip if not enough data
                print(f"Insufficient data for {symbol}")
                return False
            
            # Convert to numpy arrays
            dates = np.array(df.index.strftime('%Y-%m-%d'))
            opens = df['Open'].to_numpy()
            highs = df['High'].to_numpy()
            lows = df['Low'].to_numpy()
            closes = df['Close'].to_numpy()
            volumes = df['Volume'].to_numpy()
            
            # Calculate basic indicators
            returns = np.zeros_like(closes)
            returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
            
            # Simple Moving Averages
            sma20 = np.array([np.mean(closes[max(0, i-20):i+1]) for i in range(len(closes))])
            sma50 = np.array([np.mean(closes[max(0, i-50):i+1]) for i in range(len(closes))])
            
            # Bollinger Bands
            window = 20
            rolling_std = np.array([np.std(closes[max(0, i-window):i+1]) for i in range(len(closes))])
            bb_middle = sma20
            bb_upper = bb_middle + (rolling_std * 2)
            bb_lower = bb_middle - (rolling_std * 2)
            
            # MACD
            ema12 = np.zeros_like(closes)
            ema26 = np.zeros_like(closes)
            alpha12 = 2 / (12 + 1)
            alpha26 = 2 / (26 + 1)
            
            ema12[0] = closes[0]
            ema26[0] = closes[0]
            
            for i in range(1, len(closes)):
                ema12[i] = closes[i] * alpha12 + ema12[i-1] * (1 - alpha12)
                ema26[i] = closes[i] * alpha26 + ema26[i-1] * (1 - alpha26)
            
            macd = ema12 - ema26
            signal_line = np.array([np.mean(macd[max(0, i-9):i+1]) for i in range(len(macd))])
            
            # RSI
            diff = np.diff(closes, prepend=closes[0])
            gains = np.where(diff > 0, diff, 0)
            losses = np.where(diff < 0, -diff, 0)
            
            avg_gains = np.array([np.mean(gains[max(0, i-14):i+1]) for i in range(len(gains))])
            avg_losses = np.array([np.mean(losses[max(0, i-14):i+1]) for i in range(len(losses))])
            
            rs = avg_gains / (avg_losses + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # Save to CSV
            file_path = os.path.join(self.data_dir, f'{symbol}_data.csv')
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                               'Returns', 'SMA20', 'SMA50', 'BB_Upper', 'BB_Lower',
                               'BB_Middle', 'MACD', 'Signal_Line', 'RSI'])
                
                for i in range(len(dates)):
                    writer.writerow([
                        dates[i], opens[i], highs[i], lows[i], closes[i],
                        volumes[i], returns[i], sma20[i], sma50[i],
                        bb_upper[i], bb_lower[i], bb_middle[i],
                        macd[i], signal_line[i], rsi[i]
                    ])
            
            print(f"Successfully downloaded and processed {symbol} data")
            return True
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            return False

    def collect_all_data(self):
        """Download data for all symbols"""
        success_count = 0
        for symbol in self.symbols:
            if self.download_stock_data(symbol):
                success_count += 1
        
        print(f"\nSuccessfully collected data for {success_count} out of {len(self.symbols)} stocks")

if __name__ == "__main__":
    collector = StockDataCollector()
    collector.collect_all_data()
