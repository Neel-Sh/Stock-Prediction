# Stock Price Prediction with LSTM Neural Network

A simplified two-file system for training LSTM neural networks to predict stock prices with epoch-based training and accuracy visualization.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Usage

**Step 1: Train a Model**
```bash
python train_model.py
```
- Downloads 5 years of stock data
- Engineers 20+ technical indicators
- Trains LSTM neural network for 50 epochs
- Shows training curves and accuracy plots
- Saves trained model and scaler

**Step 2: Make Real-Time Predictions**
```bash
python predict_realtime.py
```
- Loads your trained model
- Downloads recent data
- Makes 5-day predictions
- Provides buy/sell/hold recommendations
- Shows prediction chart

## 📊 Features

### 🧠 Deep Learning Model
- **LSTM Neural Network**: 3-layer LSTM with dropout
- **Sequence Learning**: Uses 60 days to predict next day
- **Epoch Training**: Shows training progress over 50 epochs
- **Validation**: 80/20 train-test split with time series validation

### 📈 Technical Indicators
- **Moving Averages**: SMA and EMA (5, 10, 20, 50 days)
- **Momentum**: RSI, MACD, MACD Signal
- **Volatility**: Bollinger Bands, Price volatility
- **Volume**: Volume ratios and trends
- **Price Action**: Returns, high-low percentages

### 📊 Visualization
- **Training Curves**: Loss over epochs
- **Actual vs Predicted**: Full time series comparison
- **Recent Performance**: Last 100 days zoom-in
- **Error Distribution**: Prediction accuracy analysis
- **Future Predictions**: 5-day forecast chart

### 🎯 Smart Recommendations
- **Buy/Sell/Hold**: Based on predicted price movement
- **Confidence Scoring**: Multi-factor confidence calculation
- **Technical Analysis**: RSI, Bollinger Bands, Volume confirmation
- **Risk Assessment**: Accuracy metrics and error bounds

## 📁 Output Files

### Training (`train_model.py`)
- `{SYMBOL}_lstm_model.h5` - Trained LSTM model
- `{SYMBOL}_scaler.pkl` - Feature scaler for predictions
- `model_analysis.png` - Training and accuracy plots

### Prediction (`predict_realtime.py`)
- `{SYMBOL}_prediction.png` - Future price prediction chart
- Console output with recommendations and confidence

## 🔧 How It Works

### Training Process
1. **Data Collection**: Downloads 5 years of stock data directly from Yahoo Finance
2. **Feature Engineering**: Creates 20+ technical indicators
3. **Sequence Preparation**: Creates 60-day sequences for LSTM training
4. **Model Training**: Trains 3-layer LSTM for 50 epochs
5. **Evaluation**: Tests on unseen data and creates accuracy plots
6. **Model Saving**: Saves trained model and scaler for predictions

### Prediction Process
1. **Model Loading**: Loads trained LSTM model and scaler
2. **Recent Data**: Downloads last 6 months of data
3. **Feature Engineering**: Applies same features as training
4. **Prediction**: Forecasts next 5 days of prices
5. **Analysis**: Generates buy/sell recommendations
6. **Visualization**: Creates prediction chart

## 📊 Model Architecture

```
Input: 60 days × 20 features
       ↓
LSTM Layer (100 units) + Dropout
       ↓
LSTM Layer (100 units) + Dropout
       ↓
LSTM Layer (50 units) + Dropout
       ↓
Dense Layer (25 units)
       ↓
Output: Next day price
```

## 🎯 Accuracy Metrics

The system provides multiple accuracy measures:
- **MSE/MAE/RMSE**: Standard regression metrics
- **Percentage Accuracy**: Within 1% and 5% thresholds
- **Directional Accuracy**: Predicting up/down movement
- **Confidence Scoring**: Multi-factor confidence assessment

## 📈 Example Output

```
🎯 RECOMMENDATION FOR AAPL
==================================================
🚀 STRONG BUY
💰 Current Price: $185.25
🎯 Target Price: $195.80
📊 Expected Return: +5.69%
🎲 Confidence: 78.5%

📊 Technical Indicators:
   RSI: 45.2 (Neutral)
   BB Position: 0.65 (Middle Range)
   Volume Ratio: 1.34x (Normal)
```

## ⚠️ Requirements

- Python 3.8+
- TensorFlow 2.13+
- Internet connection for data download
- Sufficient disk space for model files

## 🔬 Technical Details

### Data Source
- **Yahoo Finance API**: Direct HTTP requests (no API key needed)
- **5-Year History**: Comprehensive training data
- **Daily Resolution**: End-of-day prices and volumes

### Feature Engineering
- **Price Features**: Returns, volatility, price changes
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Moving Averages**: Multiple timeframes
- **Volume Analysis**: Volume ratios and trends
- **Normalization**: MinMax scaling for neural network

### Model Training
- **Sequence Length**: 60 days
- **Batch Size**: 32
- **Epochs**: 50
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error
- **Validation**: 20% holdout

## 📋 Troubleshooting

### Common Issues

1. **TensorFlow Installation**
   ```bash
   pip install tensorflow>=2.13.0
   ```

2. **Data Download Errors**
   - Check internet connection
   - Verify stock symbol is valid
   - Try again later if rate limited

3. **Model Loading Errors**
   - Ensure you've trained a model first
   - Check file permissions
   - Verify model files exist

4. **Memory Issues**
   - Reduce sequence length if needed
   - Use smaller batch size
   - Close other applications

## 🎓 Educational Value

This system demonstrates:
- **Time Series Prediction**: Using historical sequences
- **Deep Learning**: LSTM neural networks
- **Feature Engineering**: Technical indicator creation
- **Model Evaluation**: Comprehensive accuracy testing
- **Financial Analysis**: Stock market prediction techniques

## ⚠️ Disclaimer

This software is for educational purposes only. Stock predictions are inherently uncertain and should not be used for actual trading without proper risk management. Always consult financial advisors before making investment decisions. 