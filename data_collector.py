import numpy as np
import csv
import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from time import sleep

class StockDataCollector:
    def __init__(self):
        self.data_dir = 'stock_data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Initialize exchange symbols
        self.nasdaq_symbols = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'ADBE', 'NFLX', 'INTC',
            'AMD', 'QCOM', 'AVGO', 'CSCO', 'ORCL', 'ASML', 'TXN', 'INTU', 'AMAT', 'MU',
            'CRM', 'NOW', 'SNOW', 'WDAY', 'ZS', 'PANW', 'FTNT', 'CRWD', 'NET', 'DDOG',
            # Healthcare & Biotech
            'GILD', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'ISRG', 'MRNA', 'AMGN', 'BNTX',
            'INCY', 'SGEN', 'ALNY', 'BGNE', 'TECH', 'UTHR', 'IONS', 'BMRN', 'RARE',
            # Consumer & Retail
            'COST', 'SBUX', 'PEP', 'ABNB', 'BKNG', 'MDLZ', 'TMUS', 'PYPL', 'ADSK',
            'LULU', 'ROST', 'DLTR', 'CPRT', 'ORLY', 'AAP', 'ULTA', 'EBAY', 'ETSY',
            # Financial Technology
            'COIN', 'SQ', 'AFRM', 'SOFI', 'HOOD', 'UPST', 'MELI', 'GLBE', 'WISE',
            # Industrial & Manufacturing
            'CDNS', 'SNPS', 'ANSS', 'KLAC', 'LRCX', 'KEYS', 'TER', 'MPWR', 'SWKS',
            # Clean Energy & EVs
            'ENPH', 'SEDG', 'FSLR', 'RUN', 'PLUG', 'CHPT', 'RIVN', 'LCID'
        ]

        self.nyse_symbols = [
            # Financial Services
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'BX', 'KKR',
            'V', 'MA', 'SCHW', 'USB', 'PNC', 'TFC', 'COF', 'AIG', 'MET', 'PRU',
            # Healthcare
            'JNJ', 'PFE', 'MRK', 'ABT', 'TMO', 'DHR', 'UNH', 'CVS', 'ABBV', 'ELV',
            'HUM', 'CI', 'ZTS', 'BSX', 'BDX', 'BAX', 'SYK', 'MDT', 'LLY', 'BMY',
            # Industrial
            'HON', 'GE', 'MMM', 'CAT', 'DE', 'BA', 'LMT', 'RTX', 'NOC', 'GD',
            'EMR', 'ETN', 'CMI', 'ROK', 'PH', 'ITW', 'ROP', 'IR', 'DOV', 'FDX',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'OXY', 'PSX', 'VLO', 'MPC',
            'KMI', 'WMB', 'ET', 'EPD', 'HAL', 'BKR', 'DVN', 'HES', 'MRO', 'APA',
            # Consumer Staples
            'PG', 'KO', 'WMT', 'CL', 'PEP', 'COST', 'EL', 'PM', 'MO', 'KMB',
            'GIS', 'K', 'HSY', 'STZ', 'TAP', 'KDP', 'TSN', 'CAG', 'CPB', 'HRL',
            # Materials
            'LIN', 'APD', 'ECL', 'DD', 'NEM', 'FCX', 'DOW', 'VMC', 'MLM', 'NUE',
            'CF', 'MOS', 'ALB', 'FMC', 'EMN', 'PPG', 'SHW', 'IFF', 'CE', 'SEE',
            # Real Estate
            'PLD', 'AMT', 'CCI', 'EQIX', 'DLR', 'O', 'WY', 'SPG', 'WELL', 'VTR',
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PCG', 'XEL', 'WEC'
        ]

        self.lse_symbols = [
            # FTSE 100 Components - Financial
            'HSBA.L', 'LLOY.L', 'BARC.L', 'NWG.L', 'STAN.L', 'LGEN.L', 'PRU.L', 'AV.L',
            '3IN.L', 'PHNX.L', 'SBRY.L', 'SDR.L', 'INVP.L', 'HL.L', 'CBG.L', 'JUP.L',
            # Energy & Mining
            'BP.L', 'SHEL.L', 'RIO.L', 'AAL.L', 'GLEN.L', 'BHP.L', 'EVR.L', 'ANTO.L',
            'KAZ.L', 'TLW.L', 'CNE.L', 'PMO.L', 'ENQ.L', 'HBR.L', 'FRES.L',
            # Consumer & Retail
            'TSCO.L', 'SBRY.L', 'MKS.L', 'JD.L', 'NEXT.L', 'ABF.L', 'BRBY.L', 'KGF.L',
            'OCDO.L', 'BME.L', 'MNDI.L', 'BDEV.L', 'PSN.L', 'RMV.L', 'AUTO.L',
            # Healthcare & Pharma
            'GSK.L', 'AZN.L', 'SN.L', 'BVIC.L', 'HCM.L', 'INDV.L', 'SPI.L', 'UDG.L',
            # Technology & Telecom
            'VOD.L', 'BT-A.L', 'EXPN.L', 'REL.L', 'SGE.L', 'AUTO.L', 'SPX.L', 'AVV.L',
            # Industrial & Engineering
            'RR.L', 'BA.L', 'WEIR.L', 'IMI.L', 'SMIN.L', 'MRO.L', 'RMG.L', 'CKN.L',
            # Real Estate & Construction
            'LAND.L', 'BLND.L', 'SGRO.L', 'HMSO.L', 'DLN.L', 'BBOX.L', 'UTG.L', 'GRI.L'
        ]

        self.fse_symbols = [
            # DAX Components - Automotive
            'BMW.DE', 'VOW3.DE', 'PAH3.DE', 'POR.DE', 'CON.DE', 'SHA.DE', 'MBG.DE',
            # Technology & Software
            'SAP.DE', 'IFX.DE', 'ASME.DE', 'SOW.DE', 'UTDI.DE', 'S92.DE', 'NEM.DE',
            # Industrial & Engineering
            'SIE.DE', 'HEI.DE', 'LIN.DE', 'AIR.DE', 'MTX.DE', 'G1A.DE', 'KBX.DE',
            'ZAL.DE', 'KGX.DE', 'DHL.DE', 'RHM.DE', 'B5A.DE',
            # Financial Services
            'ALV.DE', 'DBK.DE', 'MUV2.DE', 'CBK.DE', 'DKB.DE', 'COM.DE', 'HNR1.DE',
            # Healthcare & Chemicals
            'BAS.DE', 'BAYN.DE', 'MRK.DE', 'FRE.DE', 'QIA.DE', 'SHL.DE', 'EVT.DE',
            # Energy & Utilities
            'RWE.DE', 'EOAN.DE', 'LXS.DE', 'BOSS.DE', 'UN01.DE', 'VNA.DE', 'LEG.DE',
            # Consumer & Retail
            'ADS.DE', 'HFG.DE', 'DEQ.DE', 'BEI.DE', 'CEC.DE', 'GXI.DE', 'NOEJ.DE',
            # Real Estate & Construction
            'DIC.DE', 'TEG.DE', 'LEG.DE', 'TAG.DE', 'VIB3.DE', 'AT1.DE', 'O2D.DE'
        ]

        # Combine all symbols
        self.symbols = self.nasdaq_symbols + self.nyse_symbols + self.lse_symbols + self.fse_symbols
        print(f'Total symbols to process: {len(self.symbols)}')
        print(f'NASDAQ: {len(self.nasdaq_symbols)}, NYSE: {len(self.nyse_symbols)}, '
              f'LSE: {len(self.lse_symbols)}, FSE: {len(self.fse_symbols)}')

    def download_stock_data(self, symbol, period='2y'):
        """Download stock data using yfinance and save as numpy arrays in CSV"""
        try:
            # Add delay to prevent rate limiting
            sleep(0.1)
            
            # Download data using yfinance
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            # Get fundamental data
            info = stock.info
            fundamentals = {
                'PE_Ratio': info.get('forwardPE', np.nan),
                'PB_Ratio': info.get('priceToBook', np.nan),
                'Dividend_Yield': info.get('dividendYield', np.nan),
                'Market_Cap': info.get('marketCap', np.nan),
                'Debt_To_Equity': info.get('debtToEquity', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'ROA': info.get('returnOnAssets', np.nan),
                'Quick_Ratio': info.get('quickRatio', np.nan),
                'Beta': info.get('beta', np.nan),
                'EPS_Growth': info.get('earningsQuarterlyGrowth', np.nan)
            }
            
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
                               'BB_Middle', 'MACD', 'Signal_Line', 'RSI',
                               'PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'Market_Cap',
                               'Debt_To_Equity', 'ROE', 'ROA', 'Quick_Ratio', 'Beta',
                               'EPS_Growth'])
                
                for i in range(len(dates)):
                    writer.writerow([
                        dates[i], opens[i], highs[i], lows[i], closes[i],
                        volumes[i], returns[i], sma20[i], sma50[i],
                        bb_upper[i], bb_lower[i], bb_middle[i],
                        macd[i], signal_line[i], rsi[i],
                        fundamentals['PE_Ratio'], fundamentals['PB_Ratio'],
                        fundamentals['Dividend_Yield'], fundamentals['Market_Cap'],
                        fundamentals['Debt_To_Equity'], fundamentals['ROE'],
                        fundamentals['ROA'], fundamentals['Quick_Ratio'],
                        fundamentals['Beta'], fundamentals['EPS_Growth']
                    ])
            
            print(f"Successfully downloaded and processed {symbol} data")
            return True
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            return False

    def collect_all_data(self):
        """Download data for all symbols using parallel processing"""
        print("Starting data collection...")
        success_count = 0
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(self.download_stock_data, self.symbols))
            success_count = sum(1 for r in results if r)
        
        print(f"\nSuccessfully collected data for {success_count} out of {len(self.symbols)} stocks")
        
        # Print exchange-specific stats
        nasdaq_success = sum(1 for s, r in zip(self.symbols, results) 
                            if r and s in self.nasdaq_symbols)
        nyse_success = sum(1 for s, r in zip(self.symbols, results) 
                          if r and s in self.nyse_symbols)
        lse_success = sum(1 for s, r in zip(self.symbols, results) 
                         if r and s in self.lse_symbols)
        fse_success = sum(1 for s, r in zip(self.symbols, results) 
                         if r and s in self.fse_symbols)
        
        print(f"NASDAQ: {nasdaq_success}/{len(self.nasdaq_symbols)} stocks")
        print(f"NYSE: {nyse_success}/{len(self.nyse_symbols)} stocks")
        print(f"LSE: {lse_success}/{len(self.lse_symbols)} stocks")
        print(f"FSE: {fse_success}/{len(self.fse_symbols)} stocks")

if __name__ == "__main__":
    collector = StockDataCollector()
    collector.collect_all_data()
