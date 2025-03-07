# Stock Market Prediction System

A machine learning-based system for predicting stock market movements using technical indicators and ensemble learning. The system uses NumPy for efficient data processing and combines multiple models for robust predictions.

## Features

- Data processing using pure NumPy (no pandas dependency)
- Technical indicators calculation:
  - Simple Moving Averages (20 and 50-day)
  - Relative Strength Index (RSI)
  - Bollinger Bands
- Ensemble model combining:
  - Random Forest
  - XGBoost
  - Gradient Boosting
- Real-time stock data fetching using yfinance
- Normalized feature engineering
- Support for multiple stocks

## Project Structure

```
.
├── README.md
├── requirements.txt
├── stock_model.py      # Core model implementation
├── predict_stocks.py   # Prediction script
├── stock_data/        # Directory for processed stock data
└── trained_models/    # Directory for saved models
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train a new model:

```bash
python stock_model.py
```

This will:
1. Download historical data for configured stocks
2. Process and calculate technical indicators
3. Train an ensemble model
4. Save the model to `trained_models/unified_model.joblib`

### Making Predictions

To get predictions for stocks:

```bash
python predict_stocks.py
```

This will:
1. Fetch current market data for configured stocks
2. Calculate technical indicators
3. Make predictions using the trained model
4. Display top 5 stocks with highest upward movement probability

## Model Details

The system uses an ensemble of three models:

1. **Random Forest Classifier**
   - 1000 estimators
   - Max depth: 10
   - Balanced class weights

2. **XGBoost Classifier**
   - 1000 estimators
   - Learning rate: 0.005
   - Max depth: 7

3. **Gradient Boosting Classifier**
   - 1000 estimators
   - Learning rate: 0.005
   - Max depth: 7

Features used for prediction:
- Closing prices (normalized)
- Trading volumes (normalized)
- Price returns
- Technical indicators:
  - SMA20 and SMA50 distances
  - RSI
  - Volume trends
  - Price momentum
  - Bollinger Bands positions
  - Volatility

## Output Format

The prediction script outputs:
- Company name and symbol
- Current stock price
- Market capitalization
- P/E ratio
- Trading volume
- Upward movement probability

Example output:
```
1. NVIDIA Corporation (NVDA)
   Sector: Technology
   Current Price: $110.57
   Market Cap: $2,697,907,929,088
   P/E Ratio: 26.84
   Daily Volume: 319,708,800
   Upward Movement Probability: 5.3%
```

## Dependencies

- numpy
- scikit-learn
- xgboost
- yfinance
- joblib

## Notes

- The system uses pure NumPy for calculations to minimize dependencies
- All features are normalized before prediction
- The model is trained on historical data from major tech stocks
- Predictions are based on technical indicators and do not consider fundamental analysis

## Disclaimer

This system is for educational purposes only. Stock market predictions are inherently uncertain, and you should not make investment decisions solely based on this system's output.
