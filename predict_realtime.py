import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.model = None
        self.scaler = None
        self.market_scaler = None
        self.ensemble_models = []
        self.feature_names = None
        self.metadata = None
        self.sequence_length = 60
        self.market_data = None
        
    def load_model(self):
        """Load the trained advanced model and ensemble"""
        import os
        
        # Try loading advanced model first
        model_filename = f'{self.symbol}_advanced_lstm_model.h5'
        scaler_filename = f'{self.symbol}_scaler.pkl'
        features_filename = f'{self.symbol}_features.pkl'
        metadata_filename = f'{self.symbol}_metadata.pkl'
        
        try:
            print(f"📂 Loading advanced model for {self.symbol}...")
            
            # Load primary model
            if os.path.exists(model_filename):
                self.model = tf.keras.models.load_model(model_filename)
                print("✅ Primary advanced model loaded!")
            else:
                # Fallback to old model format
                old_model_filename = f'{self.symbol}_lstm_model.h5'
                if os.path.exists(old_model_filename):
                    self.model = tf.keras.models.load_model(old_model_filename)
                    print("✅ Legacy model loaded!")
                else:
                    raise FileNotFoundError("No model file found")
            
            # Load scaler
            with open(scaler_filename, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Scaler loaded!")
            
            # Load feature names if available
            if os.path.exists(features_filename):
                with open(features_filename, 'rb') as f:
                    self.feature_names = pickle.load(f)
                print(f"✅ Feature names loaded ({len(self.feature_names)} features)!")
            else:
                print("⚠️ Feature names not found, using default feature set")
                self.feature_names = None
            
            # Load metadata if available
            if os.path.exists(metadata_filename):
                with open(metadata_filename, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"✅ Model metadata loaded (Type: {self.metadata.get('model_type', 'Unknown')})!")
            else:
                self.metadata = None
            
            # Load ensemble models if available
            self.ensemble_models = []
            ensemble_pattern = f'{self.symbol}_ensemble_'
            ensemble_files = [f for f in os.listdir('.') if f.startswith(ensemble_pattern) and f.endswith('.h5')]
            
            for ensemble_file in ensemble_files:
                try:
                    ensemble_model = tf.keras.models.load_model(ensemble_file)
                    model_name = ensemble_file.replace(ensemble_pattern, '').replace('.h5', '')
                    self.ensemble_models.append((model_name, ensemble_model))
                    print(f"✅ Ensemble model loaded: {model_name}")
                except Exception as e:
                    print(f"⚠️ Could not load ensemble model {ensemble_file}: {e}")
            
            if self.ensemble_models:
                print(f"🎯 Total ensemble models: {len(self.ensemble_models)}")
            
            # Load market scaler if available
            market_scaler_filename = f'{self.symbol}_market_scaler.pkl'
            if os.path.exists(market_scaler_filename):
                with open(market_scaler_filename, 'rb') as f:
                    self.market_scaler = pickle.load(f)
                print("✅ Market scaler loaded!")
            else:
                self.market_scaler = None
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Model files not found for {self.symbol}")
            print(f"   Please run 'python train_model.py' first to train the model")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def download_stock_data(self, symbol, start_date, end_date):
        """Download stock data directly from Yahoo Finance API"""
        print(f"📊 Downloading {symbol} data from {start_date} to {end_date}...")
        
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            'period1': start_ts,
            'period2': end_ts,
            'interval': '1d',
            'includePrePost': 'true',
            'events': 'div%2Csplit'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, params=params, headers=headers)
            data = response.json()
            
            if 'chart' not in data or not data['chart']['result']:
                print(f"❌ No data found for {symbol}")
                return None
                
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            # Create DataFrame
            df = pd.DataFrame({
                'Date': [datetime.fromtimestamp(ts) for ts in timestamps],
                'Open': quotes['open'],
                'High': quotes['high'],
                'Low': quotes['low'],
                'Close': quotes['close'],
                'Volume': quotes['volume']
            })
            
            # Remove rows with missing data
            df = df.dropna()
            df = df.reset_index(drop=True)
            
            print(f"✅ Downloaded {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {str(e)}")
            return None
    
    def download_market_context(self, start_date, end_date):
        """Download S&P 500 and VIX data for market context"""
        print("📊 Downloading market context data...")
        
        # Download S&P 500
        spy_data = self.download_stock_data('^GSPC', start_date, end_date)
        vix_data = self.download_stock_data('^VIX', start_date, end_date)
        
        if spy_data is None or vix_data is None:
            print("⚠️ Warning: Could not download market context data")
            return None
        
        # Combine market data
        market_df = pd.DataFrame({
            'Date': spy_data['Date'],
            'SPY_Close': spy_data['Close'],
            'SPY_Volume': spy_data['Volume'],
            'SPY_Returns': spy_data['Close'].pct_change(),
            'VIX_Close': vix_data['Close'].reindex(spy_data.index),
        })
        
        # Add market features
        market_df['SPY_MA_20'] = market_df['SPY_Close'].rolling(20).mean()
        market_df['SPY_Volatility'] = market_df['SPY_Returns'].rolling(20).std()
        market_df['Market_Trend'] = (market_df['SPY_Close'] > market_df['SPY_MA_20']).astype(int)
        market_df['Fear_Greed'] = 1 / (1 + market_df['VIX_Close'] / 20)  # Normalized fear/greed
        
        market_df = market_df.dropna()
        self.market_data = market_df
        print(f"✅ Market context data prepared: {len(market_df)} records")
        return market_df
    
    def engineer_features(self, df):
        """Create advanced technical indicators and features (same as training)"""
        print("🔧 Engineering advanced features...")
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Change'] = df['Close'] - df['Open']
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']
        df['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        
        # Multiple timeframe moving averages
        ma_periods = [3, 5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 144, 200]
        for window in ma_periods:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
            df[f'Price_vs_SMA_{window}'] = df['Close'] / df[f'SMA_{window}'] - 1
            df[f'Price_vs_EMA_{window}'] = df['Close'] / df[f'EMA_{window}'] - 1
        
        # Advanced momentum indicators
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['RSI_14'] = calculate_rsi(df['Close'], 14)
        df['RSI_7'] = calculate_rsi(df['Close'], 7)
        df['RSI_21'] = calculate_rsi(df['Close'], 21)
        
        # Stochastic Oscillator
        def stochastic_oscillator(high, low, close, k_window=14, d_window=3):
            lowest_low = low.rolling(window=k_window).min()
            highest_high = high.rolling(window=k_window).max()
            k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
            d_percent = k_percent.rolling(window=d_window).mean()
            return k_percent, d_percent
        
        df['Stoch_K'], df['Stoch_D'] = stochastic_oscillator(df['High'], df['Low'], df['Close'])
        
        # Williams %R
        def williams_r(high, low, close, window=14):
            highest_high = high.rolling(window=window).max()
            lowest_low = low.rolling(window=window).min()
            return ((highest_high - close) / (highest_high - lowest_low)) * -100
        
        df['Williams_R'] = williams_r(df['High'], df['Low'], df['Close'])
        
        # MACD variations
        def calculate_macd(prices, fast=12, slow=26, signal=9):
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            return macd, signal_line, histogram
        
        df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'])
        
        # Bollinger Bands with multiple deviations
        for period in [10, 20, 50]:
            for std_dev in [1.5, 2.0, 2.5]:
                sma = df['Close'].rolling(window=period).mean()
                std = df['Close'].rolling(window=period).std()
                df[f'BB_Upper_{period}_{std_dev}'] = sma + (std * std_dev)
                df[f'BB_Lower_{period}_{std_dev}'] = sma - (std * std_dev)
                df[f'BB_Position_{period}_{std_dev}'] = (df['Close'] - df[f'BB_Lower_{period}_{std_dev}']) / (df[f'BB_Upper_{period}_{std_dev}'] - df[f'BB_Lower_{period}_{std_dev}'])
        
        # Average True Range (ATR)
        def calculate_atr(high, low, close, window=14):
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return true_range.rolling(window=window).mean()
        
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        # Commodity Channel Index (CCI)
        def calculate_cci(high, low, close, window=20):
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=window).mean()
            mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
            return (tp - sma_tp) / (0.015 * mad)
        
        df['CCI'] = calculate_cci(df['High'], df['Low'], df['Close'])
        
        # Advanced volume indicators
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio_10'] = df['Volume'] / df['Volume_SMA_10']
        df['Volume_Ratio_20'] = df['Volume'] / df['Volume_SMA_20']
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Returns']) * df['Volume']).cumsum()
        df['OBV_MA'] = df['OBV'].rolling(window=20).mean()
        
        # Price-Volume Trend (PVT)
        df['PVT'] = (df['Returns'] * df['Volume']).cumsum()
        df['PVT_MA'] = df['PVT'].rolling(window=20).mean()
        
        # Advanced volatility measures
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        df['Volatility_50'] = df['Returns'].rolling(window=50).std()
        
        # Parkinson volatility (using high-low)
        df['Parkinson_Vol'] = np.sqrt((1/(4*np.log(2))) * np.log(df['High']/df['Low'])**2).rolling(window=20).mean()
        
        # Support and Resistance levels
        df['Support_20'] = df['Low'].rolling(window=20).min()
        df['Resistance_20'] = df['High'].rolling(window=20).max()
        df['Support_Distance'] = (df['Close'] - df['Support_20']) / df['Close']
        df['Resistance_Distance'] = (df['Resistance_20'] - df['Close']) / df['Close']
        
        # Trend indicators
        for window in [5, 10, 20, 50]:
            df[f'Trend_{window}'] = (df['Close'] > df['Close'].shift(window)).astype(int)
            df[f'Higher_Highs_{window}'] = (df['High'] > df['High'].shift(window)).astype(int)
            df[f'Higher_Lows_{window}'] = (df['Low'] > df['Low'].shift(window)).astype(int)
        
        # Fibonacci retracement levels
        high_52w = df['High'].rolling(window=252).max()
        low_52w = df['Low'].rolling(window=252).min()
        df['Fib_23.6'] = low_52w + 0.236 * (high_52w - low_52w)
        df['Fib_38.2'] = low_52w + 0.382 * (high_52w - low_52w)
        df['Fib_50.0'] = low_52w + 0.500 * (high_52w - low_52w)
        df['Fib_61.8'] = low_52w + 0.618 * (high_52w - low_52w)
        
        # Merge with market context data if available
        if self.market_data is not None:
            df = df.merge(self.market_data, on='Date', how='left')
            
            # Relative performance vs market
            if 'SPY_Returns' in df.columns:
                df['Beta_20'] = df['Returns'].rolling(20).cov(df['SPY_Returns']) / df['SPY_Returns'].rolling(20).var()
                df['Alpha_20'] = df['Returns'].rolling(20).mean() - df['Beta_20'] * df['SPY_Returns'].rolling(20).mean()
                df['Relative_Strength'] = df['Returns'].rolling(20).mean() / df['SPY_Returns'].rolling(20).mean()
        
        # Lag features for temporal patterns
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        # Remove NaN values
        initial_length = len(df)
        df = df.dropna().reset_index(drop=True)
        final_length = len(df)
        
        print(f"✅ Advanced features engineered. Dataset: {initial_length} → {final_length} records")
        return df
    
    def prepare_prediction_data(self, df):
        """Prepare data for prediction using same features as training"""
        print("🔄 Preparing data for advanced prediction...")
        
        # Use stored feature names if available
        if self.feature_names is not None:
            feature_cols = self.feature_names
            print(f"📊 Using stored feature names ({len(feature_cols)} features)")
        else:
            # Fallback to automatic feature selection
            exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            print(f"⚠️ Using automatic feature selection ({len(feature_cols)} features)")
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            print(f"⚠️ Warning: Missing {len(missing)} features: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        
        if len(available_features) == 0:
            print("❌ No valid features found!")
            return None, None
        
        # Prepare features
        features = df[available_features].values
        
        # Handle any remaining NaN/inf values
        features = np.nan_to_num(features, nan=0, posinf=1e6, neginf=-1e6)
        
        # Scale the features using the saved scaler
        try:
            features_scaled = self.scaler.transform(features)
        except ValueError as e:
            print(f"❌ Scaling error: {e}")
            print(f"Expected {self.scaler.n_features_in_} features, got {features.shape[1]}")
            return None, None
        
        # Create sequence for prediction (last sequence_length days)
        if len(features_scaled) >= self.sequence_length:
            X = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, len(available_features))
            print(f"✅ Prediction sequence prepared: {X.shape}")
            return X, df
        else:
            print(f"❌ Not enough data for prediction. Need at least {self.sequence_length} days, got {len(features_scaled)}")
            return None, None
    
    def make_prediction(self, days_ahead=5):
        """Make advanced stock price predictions with ensemble"""
        print(f"🔮 Making advanced predictions for {self.symbol}...")
        
        # Download recent data (last 8 months for advanced features)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=240)).strftime('%Y-%m-%d')
        
        # Download market context data first
        print("🌍 Downloading market context...")
        market_data = self.download_market_context(start_date, end_date)
        
        # Download stock data
        print(f"📊 Downloading {self.symbol} data...")
        df = self.download_stock_data(self.symbol, start_date, end_date)
        if df is None:
            return None, None
        
        # Engineer advanced features
        df = self.engineer_features(df)
        
        # Prepare data for prediction
        X, processed_df = self.prepare_prediction_data(df)
        if X is None:
            return None, None
        
        # Make ensemble predictions
        if self.ensemble_models and len(self.ensemble_models) > 0:
            print(f"🎯 Using ensemble of {len(self.ensemble_models)} models...")
            
            ensemble_predictions = []
            model_names = []
            
            # Get predictions from each model in ensemble
            for name, model in self.ensemble_models:
                try:
                    model_preds = []
                    current_sequence = X[0].copy()
                    
                    for day in range(days_ahead):
                        pred = model.predict(current_sequence.reshape(1, self.sequence_length, current_sequence.shape[1]), verbose=0)
                        model_preds.append(pred[0][0])
                        
                        if day < days_ahead - 1:
                            # Update sequence for next prediction
                            new_row = current_sequence[-1].copy()
                            # Update key features with predicted values
                            new_row[0] = pred[0][0] if current_sequence.shape[1] > 0 else pred[0][0]
                            current_sequence = np.vstack([current_sequence[1:], new_row])
                    
                    ensemble_predictions.append(model_preds)
                    model_names.append(name)
                    print(f"   ✅ {name}: ${model_preds[-1]:.2f}")
                    
                except Exception as e:
                    print(f"   ⚠️ Error with {name}: {e}")
            
            if ensemble_predictions:
                # Calculate weighted ensemble prediction
                ensemble_array = np.array(ensemble_predictions)
                
                # Give more weight to the primary model
                weights = np.ones(len(ensemble_predictions))
                if len(ensemble_predictions) > 1:
                    weights[0] = 1.5  # Primary model gets 50% more weight
                weights = weights / np.sum(weights)
                
                # Weighted average
                predictions = np.average(ensemble_array, axis=0, weights=weights).tolist()
                
                # Calculate ensemble confidence based on agreement
                ensemble_std = np.std(ensemble_array, axis=0)
                avg_std = np.mean(ensemble_std)
                ensemble_confidence = max(0, 1 - (avg_std / np.mean(predictions))) * 100
                
                print(f"🎯 Ensemble Agreement: {ensemble_confidence:.1f}%")
            else:
                print("⚠️ Ensemble failed, using primary model...")
                predictions = self._single_model_prediction(X, days_ahead)
        else:
            print("📊 Using single model prediction...")
            predictions = self._single_model_prediction(X, days_ahead)
        
        current_price = processed_df['Close'].iloc[-1]
        print(f"💰 Current Price: ${current_price:.2f}")
        
        # Calculate prediction statistics
        pred_change = predictions[-1] - current_price
        pred_change_pct = (pred_change / current_price) * 100
        
        print(f"📈 Predicted Price ({days_ahead} days): ${predictions[-1]:.2f}")
        print(f"📊 Expected Change: ${pred_change:+.2f} ({pred_change_pct:+.2f}%)")
        
        return predictions, processed_df
    
    def _single_model_prediction(self, X, days_ahead):
        """Helper method for single model prediction"""
        predictions = []
        current_sequence = X[0].copy()
        
        for day in range(days_ahead):
            pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, current_sequence.shape[1]), verbose=0)
            predictions.append(pred[0][0])
            
            if day < days_ahead - 1:
                new_row = current_sequence[-1].copy()
                new_row[0] = pred[0][0]
                current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return predictions
    
    def generate_recommendation(self, predictions, df):
        """Generate advanced buy/sell/hold recommendation"""
        if not predictions or df is None:
            return
        
        current_price = df['Close'].iloc[-1]
        predicted_price = predictions[-1]
        change_pct = ((predicted_price - current_price) / current_price) * 100
        
        # Get recent indicators for analysis (with fallbacks for missing features)
        latest = df.iloc[-1]
        
        # Primary indicators
        rsi = latest.get('RSI_14', latest.get('RSI', 50))  # Fallback to RSI if RSI_14 not available
        
        # Bollinger Bands (try different variations)
        bb_position = None
        for period in [20, 10, 50]:
            for std_dev in [2.0, 1.5, 2.5]:
                bb_col = f'BB_Position_{period}_{std_dev}'
                if bb_col in latest.index:
                    bb_position = latest[bb_col]
                    break
            if bb_position is not None:
                break
        
        if bb_position is None:
            bb_position = latest.get('BB_Position', 0.5)  # Fallback
        
        # Volume indicators
        volume_ratio = latest.get('Volume_Ratio_20', latest.get('Volume_Ratio_10', latest.get('Volume_Ratio', 1.0)))
        
        # Additional advanced indicators
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        macd_histogram = latest.get('MACD_Histogram', macd - macd_signal)
        atr_ratio = latest.get('ATR_Ratio', 0.02)
        volatility = latest.get('Volatility_20', latest.get('Volatility', 0.02))
        
        # Market context indicators (if available)
        market_trend = latest.get('Market_Trend', 1)
        fear_greed = latest.get('Fear_Greed', 0.5)
        relative_strength = latest.get('Relative_Strength', 1.0)
        
        # Calculate confidence based on multiple advanced factors
        confidence_factors = []
        
        # 1. RSI-based confidence
        if 30 <= rsi <= 70:
            confidence_factors.append(0.8)
        elif 20 <= rsi <= 80:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # 2. Bollinger Band position confidence
        if 0.2 <= bb_position <= 0.8:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        # 3. Volume confirmation
        if volume_ratio > 1.2:
            confidence_factors.append(0.9)
        elif volume_ratio > 0.8:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        # 4. MACD confirmation
        if (macd > macd_signal and change_pct > 0) or (macd < macd_signal and change_pct < 0):
            confidence_factors.append(0.8)  # MACD agrees with prediction
        else:
            confidence_factors.append(0.6)
        
        # 5. Volatility consideration
        if atr_ratio < 0.03:  # Low volatility = higher confidence
            confidence_factors.append(0.8)
        elif atr_ratio < 0.05:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.6)
        
        # 6. Market context
        if market_trend == 1 and change_pct > 0:  # Bull market + positive prediction
            confidence_factors.append(0.9)
        elif market_trend == 0 and change_pct < 0:  # Bear market + negative prediction
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.7)
        
        # 7. Ensemble confidence (if available)
        if hasattr(self, 'ensemble_models') and self.ensemble_models:
            confidence_factors.append(0.85)  # Ensemble model bonus
        
        confidence = np.mean(confidence_factors) * 100
        
        # Generate recommendation with more nuanced thresholds
        print("\n" + "="*60)
        print(f"🎯 ADVANCED RECOMMENDATION FOR {self.symbol}")
        print("="*60)
        
        if change_pct > 7:
            recommendation = "🟢 STRONG BUY"
            emoji = "🚀"
        elif change_pct > 3:
            recommendation = "🟢 BUY"
            emoji = "📈"
        elif change_pct > -1:
            recommendation = "🟡 HOLD"
            emoji = "⚖️"
        elif change_pct > -4:
            recommendation = "🔴 SELL"
            emoji = "📉"
        else:
            recommendation = "🔴 STRONG SELL"
            emoji = "⚠️"
        
        print(f"{emoji} {recommendation}")
        print(f"💰 Current Price: ${current_price:.2f}")
        print(f"🎯 Target Price: ${predicted_price:.2f}")
        print(f"📊 Expected Return: {change_pct:+.2f}%")
        print(f"🎲 Confidence: {confidence:.1f}%")
        
        # Model information
        if hasattr(self, 'metadata') and self.metadata:
            print(f"🤖 Model Type: {self.metadata.get('model_type', 'Unknown')}")
            if hasattr(self, 'ensemble_models') and self.ensemble_models:
                print(f"🎯 Ensemble Models: {len(self.ensemble_models)}")
        
        # Advanced technical indicators summary
        print(f"\n📊 Advanced Technical Analysis:")
        print(f"   RSI (14): {rsi:.1f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})")
        print(f"   MACD: {macd:.3f} {'📈' if macd > macd_signal else '📉'} Signal: {macd_signal:.3f}")
        print(f"   BB Position: {bb_position:.2f} ({'Lower' if bb_position < 0.2 else 'Upper' if bb_position > 0.8 else 'Middle'})")
        print(f"   Volume Ratio: {volume_ratio:.2f}x {'(High)' if volume_ratio > 1.5 else '(Normal)' if volume_ratio > 0.8 else '(Low)'}")
        print(f"   Volatility: {volatility*100:.1f}% {'(High)' if volatility > 0.03 else '(Normal)' if volatility > 0.015 else '(Low)'}")
        
        # Market context (if available)
        if 'SPY_Returns' in df.columns:
            print(f"\n🌍 Market Context:")
            print(f"   Market Trend: {'Bullish' if market_trend == 1 else 'Bearish'}")
            print(f"   Fear/Greed Index: {fear_greed:.2f} ({'Greedy' if fear_greed > 0.6 else 'Fearful' if fear_greed < 0.4 else 'Neutral'})")
            if relative_strength != 1.0:
                print(f"   Relative Strength: {relative_strength:.2f}x {'(Outperforming)' if relative_strength > 1.1 else '(Underperforming)' if relative_strength < 0.9 else '(Market-like)'}")
        
        return recommendation, confidence
    
    def plot_prediction(self, predictions, df, days_ahead=5):
        """Plot historical prices and predictions"""
        print("📈 Creating prediction chart...")
        
        # Prepare data for plotting
        recent_days = min(30, len(df))  # Show last 30 days
        historical_dates = df['Date'].iloc[-recent_days:].values
        historical_prices = df['Close'].iloc[-recent_days:].values
        
        # Create future dates
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot historical prices
        plt.plot(historical_dates, historical_prices, 'b-', linewidth=2, label='Historical Price', alpha=0.8)
        
        # Plot predictions
        all_dates = list(historical_dates) + future_dates
        all_prices = list(historical_prices) + predictions
        
        # Connect last historical point to first prediction
        connection_dates = [historical_dates[-1]] + future_dates
        connection_prices = [historical_prices[-1]] + predictions
        
        plt.plot(connection_dates, connection_prices, 'r--', linewidth=2, label='Predicted Price', alpha=0.8)
        plt.plot(future_dates, predictions, 'ro', markersize=6, alpha=0.8)
        
        # Formatting
        plt.title(f'{self.symbol} Stock Price Prediction', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add annotations
        current_price = historical_prices[-1]
        final_prediction = predictions[-1]
        
        plt.annotate(f'Current: ${current_price:.2f}', 
                    xy=(historical_dates[-1], current_price),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7, edgecolor='none'),
                    color='white', fontweight='bold')
        
        plt.annotate(f'Target: ${final_prediction:.2f}', 
                    xy=(future_dates[-1], final_prediction),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7, edgecolor='none'),
                    color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Chart saved as {self.symbol}_prediction.png")

def main():
    print("🔮 Advanced Real-Time Stock Price Prediction")
    print("=" * 60)
    
    # Get stock symbol
    symbol = input("Enter stock symbol to predict (e.g., AAPL, GOOGL, MSFT): ").upper().strip()
    if not symbol:
        symbol = "AAPL"
        print(f"Using default symbol: {symbol}")
    
    # Initialize advanced predictor
    predictor = StockPredictor(symbol)
    
    # Load trained model
    print(f"\n📂 Loading models for {symbol}...")
    if not predictor.load_model():
        print(f"\n💡 To train an advanced model for {symbol}, run:")
        print(f"   python train_model.py")
        return
    
    # Display model information
    print(f"\n📊 Model Information:")
    if hasattr(predictor, 'metadata') and predictor.metadata:
        print(f"   Model Type: {predictor.metadata.get('model_type', 'Unknown')}")
        print(f"   Features: {predictor.metadata.get('num_features', 'Unknown')}")
        print(f"   Sequence Length: {predictor.metadata.get('sequence_length', 'Unknown')} days")
        print(f"   Training Date: {predictor.metadata.get('timestamp', 'Unknown')[:10]}")
    
    if hasattr(predictor, 'ensemble_models') and predictor.ensemble_models:
        print(f"   🎯 Ensemble Models: {len(predictor.ensemble_models)}")
        for i, (name, _) in enumerate(predictor.ensemble_models):
            print(f"      {i+1}. {name}")
    
    # Get prediction timeframe
    try:
        days_ahead = int(input("\nEnter prediction days ahead (1-10, default 5): ").strip() or "5")
        days_ahead = max(1, min(10, days_ahead))  # Clamp between 1 and 10
    except ValueError:
        days_ahead = 5
        print("Using default: 5 days")
    
    print(f"\n🚀 Making {days_ahead}-day prediction...")
    
    # Make advanced predictions
    predictions, df = predictor.make_prediction(days_ahead)
    
    if predictions is None:
        print("❌ Prediction failed")
        return
    
    # Generate advanced recommendation
    recommendation, confidence = predictor.generate_recommendation(predictions, df)
    
    # Display prediction summary
    current_price = df['Close'].iloc[-1]
    predicted_price = predictions[-1]
    total_return = ((predicted_price - current_price) / current_price) * 100
    daily_return = total_return / days_ahead
    
    print(f"\n📈 Prediction Summary:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   {days_ahead}-Day Target: ${predicted_price:.2f}")
    print(f"   Total Return: {total_return:+.2f}%")
    print(f"   Avg Daily Return: {daily_return:+.2f}%")
    
    # Show prediction path
    print(f"\n📅 Day-by-Day Predictions:")
    for i, pred in enumerate(predictions, 1):
        daily_change = ((pred - current_price) / current_price) * 100
        print(f"   Day {i}: ${pred:.2f} ({daily_change:+.1f}%)")
    
    # Create visualization
    print(f"\n📊 Creating prediction chart...")
    predictor.plot_prediction(predictions, df, days_ahead)
    
    # Final summary
    print(f"\n🎉 Advanced Prediction Completed for {symbol}!")
    print("=" * 60)
    print(f"📁 Files created:")
    print(f"   - {symbol}_prediction.png (prediction chart)")
    
    if hasattr(predictor, 'ensemble_models') and predictor.ensemble_models:
        print(f"\n🎯 Ensemble Prediction leveraged {len(predictor.ensemble_models)} models")
        print(f"🧠 Advanced features and market context considered")
    
    print(f"\n⚠️ Disclaimer: This is for educational purposes only.")
    print(f"   Always do your own research before making investment decisions.")

if __name__ == "__main__":
    main() 