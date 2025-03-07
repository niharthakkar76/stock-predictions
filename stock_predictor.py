import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import os
import csv
from collections import defaultdict

class StockDataCollector:
    def __init__(self):
        self.base_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'INTC',
                           'CSCO', 'CMCSA', 'PEP', 'AVGO', 'COST', 'TMUS', 'TXN', 'QCOM', 'INTU', 'AMD',
                           'PYPL', 'ABNB', 'ADP', 'AMAT', 'ADI', 'ASML', 'TEAM', 'BKNG', 'CDNS', 'CHTR',
                           'CRWD', 'DDOG', 'DXCM', 'EA', 'FTNT', 'GFS', 'IDXX', 'ILMN', 'KLAC', 'KDP',
                           'LRCX', 'LCID', 'MAR', 'MCHP', 'MDLZ', 'MNST', 'MRNA', 'MU', 'ODFL', 'PANW']
        
        # Create data directory if it doesn't exist
        self.data_dir = 'stock_data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_nasdaq_symbols(self):
        """Get list of NASDAQ symbols"""
        print("Fetching NASDAQ symbols...")
        try:
            # Start with base symbols if we can't get the full list
            return self.base_symbols
        except Exception as e:
            print(f"Error fetching NASDAQ symbols: {e}")
            print("Using base symbol list instead.")
            return self.base_symbols

    def calculate_sma(self, data, window):
        """Calculate Simple Moving Average"""
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='valid')

    def calculate_ema(self, data, span):
        """Calculate Exponential Moving Average"""
        alpha = 2 / (span + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed > 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)
        return rsi

    def download_stock_data(self, symbol, period='2y', interval='1d'):
        """Download stock data for a single symbol"""
        try:
            print(f"Downloading data for {symbol}...")
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            
            if len(data) > 0:
                # Convert to numpy arrays
                dates = np.array(data.index.astype(str))
                opens = np.array(data['Open'])
                highs = np.array(data['High'])
                lows = np.array(data['Low'])
                closes = np.array(data['Close'])
                volumes = np.array(data['Volume'])
                
                # Calculate technical indicators
                returns = np.diff(closes) / closes[:-1]
                returns = np.insert(returns, 0, 0)  # Add 0 at the beginning for alignment
                
                sma_20 = self.calculate_sma(closes, 20)
                sma_20 = np.pad(sma_20, (19, 0), mode='edge')  # Pad beginning
                
                sma_50 = self.calculate_sma(closes, 50)
                sma_50 = np.pad(sma_50, (49, 0), mode='edge')  # Pad beginning
                
                # Bollinger Bands
                bb_middle = sma_20
                rolling_std = np.array([np.std(closes[max(0, i-20):i+1]) for i in range(len(closes))])
                bb_upper = bb_middle + 2 * rolling_std
                bb_lower = bb_middle - 2 * rolling_std
                
                # MACD
                ema_12 = self.calculate_ema(closes, 12)
                ema_26 = self.calculate_ema(closes, 26)
                macd = ema_12 - ema_26
                signal_line = self.calculate_ema(macd, 9)
                
                # RSI
                rsi = self.calculate_rsi(closes)
                
                # Save to CSV
                filename = os.path.join(self.data_dir, f'{symbol}_data.csv')
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                                   'Returns', 'SMA_20', 'SMA_50', 'BB_upper', 'BB_lower',
                                   'BB_middle', 'MACD', 'Signal_Line', 'RSI'])
                    
                    for i in range(len(dates)):
                        writer.writerow([dates[i], opens[i], highs[i], lows[i], closes[i],
                                       volumes[i], returns[i], sma_20[i], sma_50[i],
                                       bb_upper[i], bb_lower[i], bb_middle[i],
                                       macd[i], signal_line[i], rsi[i]])
                
                print(f"Successfully saved data for {symbol}")
                return True
            else:
                print(f"No data available for {symbol}")
                return False
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
            return False

    def collect_all_data(self):
        """Download data for all symbols"""
        symbols = self.get_nasdaq_symbols()
        successful_downloads = 0
        
        for symbol in symbols:
            if self.download_stock_data(symbol):
                successful_downloads += 1
            time.sleep(1)  # Add delay to avoid rate limiting
            
        print(f"\nData collection completed!")
        print(f"Successfully downloaded data for {successful_downloads} out of {len(symbols)} stocks")
        print(f"Data is saved in the '{self.data_dir}' directory")

def main():
    collector = StockDataCollector()
    collector.collect_all_data()

if __name__ == "__main__":
    main()
