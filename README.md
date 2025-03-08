# Stock Trading Predictor (STP)

A sophisticated machine learning system for stock market prediction using ensemble methods and technical analysis. The system combines multiple advanced models and technical indicators to generate daily trading signals.

## Key Features

- **Real-time Data Collection**
  - Automated data fetching using yfinance
  - Support for 50+ major stocks across sectors
  - Historical and real-time price data

- **Advanced Technical Analysis**
  - Simple Moving Averages (SMA20, SMA50)
  - Relative Strength Index (RSI)
  - Bollinger Bands
  - Volume Force Index
  - Price Momentum
  - Custom momentum indicators

- **Ensemble Learning Model**
  - RandomForest (1000 trees)
  - XGBoost (1000 trees)
  - Gradient Boosting (1000 trees)
  - Voting Classifier for final predictions

## Project Structure

```
├── data_collector.py      # Stock data collection and processing
├── data_preprocessor.py   # Feature engineering and data cleaning
├── predict_stocks.py      # Main prediction interface
├── stock_model.py         # Model implementation
├── requirements.txt       # Project dependencies
├── stock_data/           # Stock data storage
├── processed_data/       # Processed features
└── trained_models/       # Saved model files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SHIV000000/spmm.git
cd spmm
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Collecting Data
```bash
python data_collector.py
```
This will:
- Fetch historical data for all configured stocks
- Calculate technical indicators
- Save processed data to stock_data directory

### Training Models
```bash
python stock_model.py
```
This will:
- Load processed stock data
- Train ensemble models with optimized parameters
- Save trained models to trained_models directory

### Making Predictions
```bash
python predict_stocks.py
```
This will:
- Load latest stock data
- Generate predictions for next 7 days
- Display results with confidence scores

## Technical Implementation

### Data Processing
- Robust preprocessing with StockDataPreprocessor
- Missing value handling with rolling window means
- Outlier removal using z-score method (threshold=3)
- Feature scaling with RobustScaler
- Data saved as SYMBOL_data.csv (e.g., AAPL_data.csv)

### Model Architecture
- **Ensemble Approach**
  - VotingClassifier with soft voting
  - Combines predictions from all base models
  - Optimized for class imbalance

- **RandomForestClassifier**
  - 1000 trees with balanced subsample weighting
  - Max depth: 10
  - Bootstrap sampling

- **XGBoostClassifier**
  - 1000 trees with class imbalance handling
  - Learning rate: 0.005
  - Max depth: 7
  - L1/L2 regularization

- **GradientBoostingClassifier**
  - 1000 trees with validation-based early stopping
  - Learning rate: 0.005
  - Subsample ratio: 0.8

### Feature Engineering
- Price-based features:
  * Close price
  * Price Range Percentage
  * Price Momentum
- Volume indicators:
  * Volume
  * Volume Force Index
  * Volume Weighted Average Price (VWAP)
- Technical indicators:
  * SMA20, SMA50
  * RSI (14-period)
  * Bollinger Bands
  * MACD with Signal Line
  * True Range
  * Custom momentum indicators

## Dependencies

- scikit-learn: Machine learning algorithms
- numpy: Numerical computations
- statsmodels: Statistical models and time series analysis
- ta-lib: Technical analysis indicators
- yfinance: Real-time stock data fetching
- xgboost: Gradient boosting implementation
- joblib: Model persistence and parallel computing
- scipy: Scientific computing and optimization

