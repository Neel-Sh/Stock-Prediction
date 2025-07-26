import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Concatenate, BatchNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class StockTrainer:
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers
        self.market_scaler = RobustScaler()  # For market context data
        self.model = None
        self.ensemble_models = []
        self.sequence_length = 60  # Use 60 days to predict next day
        self.market_data = None  # Store S&P 500 and VIX data
        
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
        print("📊 Downloading market context data (S&P 500, VIX)...")
        
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
        """Create advanced technical indicators and features"""
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
        print(f"📊 Total features: {len([col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])}")
        return df
    
    def prepare_sequences(self, data, target_col='Close'):
        """Prepare sequences for LSTM training with feature selection"""
        print("🔄 Preparing sequences for advanced LSTM...")
        
        # Exclude non-feature columns
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Remove highly correlated features to reduce overfitting
        feature_data = data[feature_cols]
        correlation_matrix = feature_data.corr().abs()
        
        # Find pairs of features with correlation > 0.95
        high_corr_var = np.where(np.triu(correlation_matrix, k=1) > 0.95)
        high_corr_var = [(correlation_matrix.index[x], correlation_matrix.columns[y]) 
                         for x, y in zip(*high_corr_var)]
        
        # Remove one feature from each highly correlated pair
        cols_to_drop = set()
        for var1, var2 in high_corr_var:
            # Keep the feature with more descriptive name (longer name often means more specific)
            if len(var1) > len(var2):
                cols_to_drop.add(var2)
            else:
                cols_to_drop.add(var1)
        
        final_features = [col for col in feature_cols if col not in cols_to_drop]
        print(f"📊 Feature selection: {len(feature_cols)} → {len(final_features)} features")
        print(f"🗑️ Removed {len(cols_to_drop)} highly correlated features")
        
        # Prepare data
        features = data[final_features].values
        target = data[target_col].values
        
        # Handle any remaining NaN/inf values
        features = np.nan_to_num(features, nan=0, posinf=1e6, neginf=-1e6)
        
        # Scale the features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target[i])
        
        X, y = np.array(X), np.array(y)
        print(f"✅ Created {len(X)} sequences of length {self.sequence_length}")
        print(f"📊 Input shape: {X.shape}, Output shape: {y.shape}")
        
        # Store feature names for later use
        self.feature_names = final_features
        
        return X, y, features_scaled
    
    def build_advanced_model(self, input_shape):
        """Build advanced LSTM model with attention and bidirectional layers"""
        print("🏗️ Building advanced LSTM model with attention...")
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First bidirectional LSTM layer
        lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputs)
        lstm1 = BatchNormalization()(lstm1)
        
        # Second bidirectional LSTM layer
        lstm2 = Bidirectional(LSTM(96, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        
        # Third bidirectional LSTM layer
        lstm3 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(lstm2)
        lstm3 = BatchNormalization()(lstm3)
        
        # Multi-head attention layer
        attention = MultiHeadAttention(num_heads=8, key_dim=32)(lstm3, lstm3)
        attention = Dropout(0.2)(attention)
        
        # Add & Norm
        attention_output = tf.keras.layers.Add()([lstm3, attention])
        attention_output = tf.keras.layers.LayerNormalization()(attention_output)
        
        # Global average pooling to reduce sequence dimension
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
        
        # Dense layers with residual connections
        dense1 = Dense(128, activation='relu')(pooled)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.3)(dense2)
        
        dense3 = Dense(32, activation='relu')(dense2)
        dense3 = Dropout(0.2)(dense3)
        
        # Output layer
        outputs = Dense(1, activation='linear')(dense3)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with custom optimizer
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )
        
        print("✅ Advanced model built successfully")
        print(f"📊 Model parameters: {model.count_params():,}")
        return model
    
    def build_ensemble_models(self, input_shape):
        """Build multiple models for ensemble"""
        print("🏗️ Building ensemble of models...")
        
        models = []
        
        # Model 1: Deep bidirectional LSTM with attention
        model1 = self.build_advanced_model(input_shape)
        models.append(('Advanced_LSTM', model1))
        
        # Model 2: Simpler but wider LSTM
        inputs2 = Input(shape=input_shape)
        lstm2_1 = LSTM(200, return_sequences=True, dropout=0.3)(inputs2)
        lstm2_2 = LSTM(150, return_sequences=True, dropout=0.3)(lstm2_1)
        lstm2_3 = LSTM(100, return_sequences=False, dropout=0.3)(lstm2_2)
        dense2_1 = Dense(100, activation='relu')(lstm2_3)
        dense2_1 = Dropout(0.4)(dense2_1)
        dense2_2 = Dense(50, activation='relu')(dense2_1)
        outputs2 = Dense(1)(dense2_2)
        
        model2 = Model(inputs=inputs2, outputs=outputs2)
        model2.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        models.append(('Wide_LSTM', model2))
        
        # Model 3: GRU-based model for diversity
        inputs3 = Input(shape=input_shape)
        gru3_1 = Bidirectional(tf.keras.layers.GRU(100, return_sequences=True, dropout=0.2))(inputs3)
        gru3_2 = Bidirectional(tf.keras.layers.GRU(80, return_sequences=True, dropout=0.2))(gru3_1)
        gru3_3 = tf.keras.layers.GRU(60, return_sequences=False, dropout=0.2)(gru3_2)
        dense3_1 = Dense(80, activation='relu')(gru3_3)
        dense3_1 = Dropout(0.3)(dense3_1)
        outputs3 = Dense(1)(dense3_1)
        
        model3 = Model(inputs=inputs3, outputs=outputs3)
        model3.compile(optimizer=Adam(0.0008), loss='mae', metrics=['mse'])
        models.append(('BiGRU', model3))
        
        print(f"✅ Built ensemble of {len(models)} models")
        return models
    
    def train_ensemble_models(self, X, y, epochs=100, validation_split=0.2):
        """Train ensemble of models with advanced techniques"""
        print(f"🚀 Training ensemble for {epochs} epochs...")
        
        # Split validation data
        val_size = int(len(X) * validation_split)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]
        
        print(f"📊 Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        # Build ensemble models
        ensemble_models = self.build_ensemble_models((X.shape[1], X.shape[2]))
        
        # Training callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train each model in ensemble
        trained_models = []
        histories = []
        
        for name, model in ensemble_models:
            print(f"\n🔄 Training {name} model...")
            
            # Custom batch size for different models
            if 'Wide' in name:
                batch_size = 16  # Larger model needs smaller batch
            elif 'GRU' in name:
                batch_size = 16  # GRU can handle larger batches
            else:
                batch_size = 16
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=1,
                shuffle=False  # Important for time series
            )
            
            trained_models.append((name, model))
            histories.append((name, history))
            
            print(f"✅ {name} training completed!")
        
        # Select best model as primary model
        best_val_loss = float('inf')
        best_model = None
        best_history = None
        
        for (name, model), (_, history) in zip(trained_models, histories):
            val_loss = min(history.history['val_loss'])
            print(f"📊 {name} best val_loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                best_history = history
                best_name = name
        
        self.model = best_model
        self.ensemble_models = trained_models
        
        print(f"🏆 Best model: {best_name} (val_loss: {best_val_loss:.6f})")
        return best_history, histories
    
    def evaluate_and_plot(self, data, X, y, history, all_histories=None):
        """Evaluate model and create advanced visualizations"""
        print("📈 Creating advanced evaluation plots...")
        
        # Make ensemble predictions
        if self.ensemble_models:
            ensemble_predictions = []
            for name, model in self.ensemble_models:
                pred = model.predict(X, verbose=0).flatten()
                ensemble_predictions.append(pred)
            
            # Average ensemble predictions
            ensemble_avg = np.mean(ensemble_predictions, axis=0)
            # Weighted ensemble (best model gets more weight)
            best_model_pred = self.model.predict(X, verbose=0).flatten()
            predictions = 0.6 * best_model_pred + 0.4 * ensemble_avg
        else:
            predictions = self.model.predict(X, verbose=0).flatten()
        
        # Calculate comprehensive metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(y))
        pred_direction = np.sign(np.diff(predictions))
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # R-squared
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"📊 Advanced Model Performance:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   Directional Accuracy: {directional_accuracy:.1f}%")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Training History Comparison
        ax1 = plt.subplot(3, 3, 1)
        if all_histories:
            for name, hist in all_histories:
                plt.plot(hist.history['loss'], label=f'{name} Training', alpha=0.7)
                plt.plot(hist.history['val_loss'], label=f'{name} Validation', alpha=0.7, linestyle='--')
        else:
            plt.plot(history.history['loss'], label='Training Loss', color='blue')
            plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.title('Training History Comparison', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Full Series Prediction
        ax2 = plt.subplot(3, 3, 2)
        dates = data['Date'].iloc[self.sequence_length:self.sequence_length + len(y)].values
        plt.plot(dates, y, label='Actual Price', color='blue', alpha=0.8, linewidth=1.5)
        plt.plot(dates, predictions, label='Predicted Price', color='red', alpha=0.8, linewidth=1.5)
        plt.title('Full Series: Actual vs Predicted', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Recent Performance (Last 100 days)
        ax3 = plt.subplot(3, 3, 3)
        recent_days = min(100, len(y))
        recent_dates = dates[-recent_days:]
        recent_actual = y[-recent_days:]
        recent_pred = predictions[-recent_days:]
        
        plt.plot(recent_dates, recent_actual, label='Actual', color='blue', linewidth=2)
        plt.plot(recent_dates, recent_pred, label='Predicted', color='red', linewidth=2)
        plt.fill_between(recent_dates, recent_actual, recent_pred, alpha=0.2, color='gray')
        plt.title(f'Recent Performance (Last {recent_days} Days)', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Error Distribution
        ax4 = plt.subplot(3, 3, 4)
        errors = y - predictions
        plt.hist(errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2)
        plt.axvline(np.mean(errors), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
        plt.title('Prediction Error Distribution', fontweight='bold')
        plt.xlabel('Error (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Scatter Plot - Actual vs Predicted
        ax5 = plt.subplot(3, 3, 5)
        plt.scatter(y, predictions, alpha=0.5, color='blue', s=10)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
        plt.title(f'Actual vs Predicted (R² = {r2:.3f})', fontweight='bold')
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Rolling Performance
        ax6 = plt.subplot(3, 3, 6)
        window = 50
        rolling_mae = pd.Series([mean_absolute_error(y[i:i+window], predictions[i:i+window]) 
                                for i in range(len(y)-window)])
        plt.plot(dates[window:], rolling_mae, color='green', linewidth=2)
        plt.title(f'Rolling MAE (Window: {window})', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('MAE')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 7: Directional Accuracy Over Time
        ax7 = plt.subplot(3, 3, 7)
        window = 30
        rolling_dir_acc = []
        for i in range(window, len(y)):
            actual_dir = np.sign(np.diff(y[i-window:i]))
            pred_dir = np.sign(np.diff(predictions[i-window:i]))
            acc = np.mean(actual_dir == pred_dir) * 100
            rolling_dir_acc.append(acc)
        
        plt.plot(dates[window:], rolling_dir_acc, color='orange', linewidth=2)
        plt.axhline(50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        plt.title(f'Rolling Directional Accuracy (Window: {window})', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 8: Feature Importance (if available)
        ax8 = plt.subplot(3, 3, 8)
        if hasattr(self, 'feature_names') and len(self.feature_names) > 0:
            # Use gradient-based feature importance
            try:
                sample_input = X[:100]  # Use first 100 samples
                with tf.GradientTape() as tape:
                    tape.watch(sample_input)
                    predictions_for_grad = self.model(sample_input)
                gradients = tape.gradient(predictions_for_grad, sample_input)
                importance = np.mean(np.abs(gradients.numpy()), axis=(0, 1))
                
                # Get top 15 features
                top_indices = np.argsort(importance)[-15:]
                top_features = [self.feature_names[i] for i in top_indices]
                top_importance = importance[top_indices]
                
                plt.barh(range(len(top_features)), top_importance, color='skyblue')
                plt.yticks(range(len(top_features)), top_features)
                plt.title('Top 15 Feature Importance', fontweight='bold')
                plt.xlabel('Importance')
            except:
                plt.text(0.5, 0.5, 'Feature importance\nnot available', 
                        ha='center', va='center', transform=ax8.transAxes)
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=ax8.transAxes)
        
        # Plot 9: Ensemble Model Comparison
        ax9 = plt.subplot(3, 3, 9)
        if self.ensemble_models:
            model_performances = []
            for name, model in self.ensemble_models:
                pred = model.predict(X, verbose=0).flatten()
                model_mae = mean_absolute_error(y, pred)
                model_performances.append((name, model_mae))
            
            names, maes = zip(*model_performances)
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            bars = plt.bar(names, maes, color=colors[:len(names)])
            plt.title('Ensemble Model Performance (MAE)', fontweight='bold')
            plt.ylabel('MAE')
            plt.xticks(rotation=45)
            
            # Highlight best model
            best_idx = np.argmin(maes)
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
        else:
            plt.text(0.5, 0.5, 'Single model\ntraining', ha='center', va='center', transform=ax9.transAxes)
        
        plt.tight_layout()
        plt.savefig('advanced_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate additional accuracy metrics
        errors = y - predictions
        accuracy_within_1pct = np.mean(np.abs(errors / y) < 0.01) * 100
        accuracy_within_5pct = np.mean(np.abs(errors / y) < 0.05) * 100
        accuracy_within_10pct = np.mean(np.abs(errors / y) < 0.10) * 100
        
        # Sharpe-like ratio for predictions
        excess_returns = np.diff(predictions)
        if np.std(excess_returns) > 0:
            prediction_sharpe = np.mean(excess_returns) / np.std(excess_returns)
        else:
            prediction_sharpe = 0
        
        print(f"\n📊 Detailed Accuracy Metrics:")
        print(f"   Within 1%: {accuracy_within_1pct:.1f}%")
        print(f"   Within 5%: {accuracy_within_5pct:.1f}%")
        print(f"   Within 10%: {accuracy_within_10pct:.1f}%")
        print(f"   Prediction Sharpe: {prediction_sharpe:.3f}")
        
        return predictions, errors
    
    def save_model(self, symbol):
        """Save the trained models and related data"""
        import pickle
        
        # Save primary model
        model_filename = f'{symbol}_advanced_lstm_model.h5'
        self.model.save(model_filename)
        print(f"✅ Primary model saved as {model_filename}")
        
        # Save ensemble models
        if self.ensemble_models:
            for i, (name, model) in enumerate(self.ensemble_models):
                ensemble_filename = f'{symbol}_ensemble_{i}_{name}.h5'
                model.save(ensemble_filename)
                print(f"✅ Ensemble model saved as {ensemble_filename}")
        
        # Save scalers
        scaler_filename = f'{symbol}_scaler.pkl'
        with open(scaler_filename, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved as {scaler_filename}")
        
        if hasattr(self, 'market_scaler') and self.market_scaler is not None:
            market_scaler_filename = f'{symbol}_market_scaler.pkl'
            with open(market_scaler_filename, 'wb') as f:
                pickle.dump(self.market_scaler, f)
            print(f"✅ Market scaler saved as {market_scaler_filename}")
        
        # Save feature names
        if hasattr(self, 'feature_names'):
            features_filename = f'{symbol}_features.pkl'
            with open(features_filename, 'wb') as f:
                pickle.dump(self.feature_names, f)
            print(f"✅ Feature names saved as {features_filename}")
        
        # Save model metadata
        metadata = {
            'symbol': symbol,
            'sequence_length': self.sequence_length,
            'num_features': len(self.feature_names) if hasattr(self, 'feature_names') else 0,
            'model_type': 'Advanced_LSTM_Ensemble',
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_filename = f'{symbol}_metadata.pkl'
        with open(metadata_filename, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Metadata saved as {metadata_filename}")

def main():
    print("🚀 Advanced Stock Price Prediction Model Training")
    print("=" * 60)
    
    # Configuration
    SYMBOL = input("Enter stock symbol (e.g., AAPL, GOOGL, MSFT): ").upper().strip()
    if not SYMBOL:
        SYMBOL = "AAPL"
        print(f"Using default symbol: {SYMBOL}")
    
    # Date range - use 7 years for better training with advanced features
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2555)).strftime('%Y-%m-%d')  # 7 years
    
    print(f"📅 Training period: {start_date} to {end_date}")
    
    # Initialize advanced trainer
    trainer = StockTrainer()
    
    # Download market context data first
    print("\n🌍 Downloading market context data...")
    market_data = trainer.download_market_context(start_date, end_date)
    
    # Download stock data
    print(f"\n📊 Downloading {SYMBOL} data...")
    df = trainer.download_stock_data(SYMBOL, start_date, end_date)
    if df is None:
        print("❌ Failed to download stock data. Exiting.")
        return
    
    # Advanced feature engineering
    print("\n🔧 Creating advanced features...")
    df = trainer.engineer_features(df)
    
    # Prepare sequences with feature selection
    print("\n🔄 Preparing sequences with feature selection...")
    X, y, features_scaled = trainer.prepare_sequences(df)
    
    if len(X) < 100:
        print("❌ Insufficient data for training. Need at least 100 sequences.")
        return
    
    # Advanced data splitting (use last 15% for testing to have more training data)
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Advanced Data Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Feature count: {X.shape[2]}")
    print(f"   Sequence length: {X.shape[1]}")
    
    # Train ensemble models with advanced techniques
    print(f"\n🚀 Training advanced ensemble models...")
    history, all_histories = trainer.train_ensemble_models(X_train, y_train, epochs=100)
    
    # Comprehensive evaluation on test set
    print(f"\n📊 Evaluating on test set...")
    
    # Test with ensemble predictions
    if trainer.ensemble_models:
        ensemble_predictions = []
        for name, model in trainer.ensemble_models:
            pred = model.predict(X_test, verbose=0).flatten()
            ensemble_predictions.append(pred)
            individual_mae = mean_absolute_error(y_test, pred)
            print(f"   {name} MAE: {individual_mae:.4f}")
        
        # Combined prediction
        ensemble_avg = np.mean(ensemble_predictions, axis=0)
        best_model_pred = trainer.model.predict(X_test, verbose=0).flatten()
        final_predictions = 0.6 * best_model_pred + 0.4 * ensemble_avg
    else:
        final_predictions = trainer.model.predict(X_test, verbose=0).flatten()
    
    # Calculate comprehensive test metrics
    test_mse = mean_squared_error(y_test, final_predictions)
    test_mae = mean_absolute_error(y_test, final_predictions)
    test_rmse = np.sqrt(test_mse)
    
    # Directional accuracy
    test_actual_direction = np.sign(np.diff(y_test))
    test_pred_direction = np.sign(np.diff(final_predictions))
    test_directional_accuracy = np.mean(test_actual_direction == test_pred_direction) * 100
    
    # R-squared
    test_ss_res = np.sum((y_test - final_predictions) ** 2)
    test_ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    test_r2 = 1 - (test_ss_res / test_ss_tot)
    
    print(f"\n🎯 Final Test Set Performance:")
    print(f"   MSE: {test_mse:.4f}")
    print(f"   MAE: {test_mae:.4f}")
    print(f"   RMSE: {test_rmse:.4f}")
    print(f"   R²: {test_r2:.4f}")
    print(f"   Directional Accuracy: {test_directional_accuracy:.1f}%")
    
    # Create comprehensive evaluation plots
    print(f"\n📈 Creating advanced visualization...")
    trainer.evaluate_and_plot(df, X_test, y_test, history, all_histories)
    
    # Save all models and data
    print(f"\n💾 Saving models and data...")
    trainer.save_model(SYMBOL)
    
    # Performance summary
    print(f"\n🎉 Advanced Training Completed Successfully!")
    print("=" * 60)
    print(f"📁 Files created:")
    print(f"   - {SYMBOL}_advanced_lstm_model.h5 (primary model)")
    if trainer.ensemble_models:
        for i, (name, _) in enumerate(trainer.ensemble_models):
            print(f"   - {SYMBOL}_ensemble_{i}_{name}.h5")
    print(f"   - {SYMBOL}_scaler.pkl (feature scaler)")
    print(f"   - {SYMBOL}_features.pkl (feature names)")
    print(f"   - {SYMBOL}_metadata.pkl (model metadata)")
    print(f"   - advanced_model_analysis.png (comprehensive analysis)")
    
    print(f"\n📊 Model Summary:")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Training Period: {start_date} to {end_date}")
    print(f"   Features: {len(trainer.feature_names) if hasattr(trainer, 'feature_names') else 'N/A'}")
    print(f"   Ensemble Models: {len(trainer.ensemble_models) if trainer.ensemble_models else 1}")
    print(f"   Final R²: {test_r2:.4f}")
    print(f"   Directional Accuracy: {test_directional_accuracy:.1f}%")

if __name__ == "__main__":
    main() 