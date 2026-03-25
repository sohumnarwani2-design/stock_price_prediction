import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OptimizedPricePredictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = None
        self.selector = None
        self.features = None
        self.best_threshold = 0.5
    
    def _clean_dataframe(self, df, symbol):
        """Clean and standardize the DataFrame columns"""
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            print(f"🔍 MultiIndex columns detected. Extracting data for {symbol}")
            try:
                # Extract data for the specific symbol
                if symbol in df.columns.get_level_values(1):
                    df_clean = df.xs(symbol, axis=1, level=1)
                    print(f"✅ Successfully extracted data for {symbol}")
                else:
                    # If symbol not found, use the first available symbol
                    first_symbol = df.columns.get_level_values(1)[0]
                    df_clean = df.xs(first_symbol, axis=1, level=1)
                    print(f"⚠️ Symbol {symbol} not found, using {first_symbol}")
            except Exception as e:
                print(f"❌ Error extracting symbol data: {e}")
                return pd.DataFrame()
        else:
            df_clean = df.copy()

        # Ensure we have the basic required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df_clean.columns:
                print(f"❌ Required column {col} not found after cleaning")
                return pd.DataFrame()

        return df_clean
    
    def prepare_features(self, df, symbol):
        """Calculate technical indicators with proper column handling"""
        try:
            # Clean the dataframe first
            df_clean = self._clean_dataframe(df, symbol)
            if df_clean.empty:
                print(f"❌ Data cleaning failed for {symbol}")
                return pd.DataFrame()

            print(f"✅ Data cleaned. Shape: {df_clean.shape}, Columns: {df_clean.columns.tolist()}")

            # Basic price features
            df_clean['Return'] = df_clean['Close'].pct_change()
            df_clean['High_Low_Ratio'] = (df_clean['High'] - df_clean['Low']) / df_clean['Close']
            df_clean['Close_Open_Ratio'] = (df_clean['Close'] - df_clean['Open']) / df_clean['Open']

            # Moving averages
            windows = [5, 10, 20]
            for window in windows:
                ma_col = f'MA{window}'
                price_to_ma_col = f'Price_to_MA{window}'

                df_clean[ma_col] = df_clean['Close'].rolling(window=window).mean()
                df_clean[price_to_ma_col] = df_clean['Close'] / df_clean[ma_col]

            # Volatility
            df_clean['Volatility_10'] = df_clean['Return'].rolling(window=10).std()

            # RSI
            delta = df_clean['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            df_clean['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df_clean['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df_clean['Close'].ewm(span=26, adjust=False).mean()
            df_clean['MACD'] = exp1 - exp2

            # Volume
            df_clean['Volume_MA20'] = df_clean['Volume'].rolling(window=20).mean()
            df_clean['Volume_Ratio'] = df_clean['Volume'] / df_clean['Volume_MA20']

            # Momentum
            df_clean['ROC_5'] = df_clean['Close'].pct_change(periods=5)
            df_clean['Momentum_5'] = df_clean['Close'] - df_clean['Close'].shift(5)

            # Target variable
            df_clean['Target'] = (df_clean['Close'].shift(-1) > df_clean['Close']).astype(int)

            result = df_clean.dropna()
            print(f"✅ Features prepared. Final shape: {result.shape}")
            return result

        except Exception as e:
            print(f"❌ Error in prepare_features for {symbol}: {e}")
            import traceback
            print(f"🔍 Full traceback:\n{traceback.format_exc()}")
            return pd.DataFrame()
    
    def train_model(self, symbol):
        """Train model for a specific symbol with detailed debugging"""
        try:
            print(f"🚀 Starting model training for {symbol}")
            
            # Download data
            print(f"📥 Downloading data for {symbol} with period: {self.config.PRICE_PREDICTION_PERIOD}")
            df = yf.download(symbol, period=self.config.PRICE_PREDICTION_PERIOD, progress=False)
            
            if df.empty:
                print(f"❌ No data downloaded for {symbol}")
                return None
            
            print(f"📊 Raw data shape: {df.shape}")
            print(f"📊 Raw data columns: {df.columns.tolist()}")
            print(f"📊 Raw data index type: {type(df.index)}")
            print(f"📊 First few rows:\n{df.head(3)}")
            
            # Check for MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                print(f"🔍 MultiIndex columns detected: {df.columns}")
                print(f"🔍 MultiIndex levels: {df.columns.levels}")
            
            # Prepare features
            print(f"🛠️ Preparing features for {symbol}...")
            df_processed = self.prepare_features(df.copy(), symbol)
            
            if df_processed.empty:
                print(f"❌ No features prepared for {symbol} after processing")
                return None
            
            print(f"✅ Features prepared successfully")
            print(f"📊 Processed data shape: {df_processed.shape}")
            print(f"📊 Processed data columns: {df_processed.columns.tolist()}")
            print(f"📊 Processed data sample:\n{df_processed.head(3)}")
            
            # Check for target variable
            if 'Target' not in df_processed.columns:
                print(f"❌ Target column not found in processed data")
                return None
            
            # Define features (excluding target and basic price columns)
            exclude_columns = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            # Also exclude any MA columns that might cause issues
            ma_columns = [col for col in df_processed.columns if col.startswith('MA')]
            exclude_columns.extend(ma_columns)
            
            features = [col for col in df_processed.columns if col not in exclude_columns]
            
            print(f"🔍 Excluded columns: {exclude_columns}")
            print(f"🎯 Features to use: {features}")
            print(f"📈 Number of features: {len(features)}")
            
            if len(features) == 0:
                print(f"❌ No features available for training")
                return None
            
            X = df_processed[features]
            y = df_processed['Target']
            
            print(f"📊 Feature matrix shape: {X.shape}")
            print(f"📊 Target vector shape: {y.shape}")
            print(f"📊 Target distribution:\n{y.value_counts()}")
            
            # Split data
            split_idx = int(len(X) * self.config.PRICE_TRAIN_TEST_SPLIT)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            print(f"📊 Training set size: {X_train.shape}")
            print(f"📊 Test set size: {X_test.shape}")
            print(f"📊 Training target distribution:\n{y_train.value_counts()}")
            print(f"📊 Test target distribution:\n{y_test.value_counts()}")
            
            # Handle class imbalance
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            scale_pos_weight = n_neg / max(n_pos, 1)  # Avoid division by zero
            
            print(f"⚖️ Class balance - Positive: {n_pos}, Negative: {n_neg}")
            print(f"⚖️ Scale positive weight: {scale_pos_weight:.2f}")
            
            # Scale features
            print(f"🔧 Scaling features...")
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            print(f"✅ Features scaled. Training shape: {X_train_scaled.shape}")
            
            # Feature selection
            n_features = min(15, len(features))
            print(f"🔍 Selecting {n_features} best features from {len(features)}...")
            
            self.selector = SelectKBest(mutual_info_classif, k=n_features)
            X_train_selected = self.selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.selector.transform(X_test_scaled)
            
            # Get selected feature names
            selected_indices = self.selector.get_support(indices=True)
            self.features = [features[i] for i in selected_indices]
            
            print(f"✅ Feature selection completed")
            print(f"🎯 Selected features: {self.features}")
            print(f"📊 Training features after selection: {X_train_selected.shape}")
            print(f"📊 Test features after selection: {X_test_selected.shape}")
            
            # Train model with simplified hyperparameters
            param_grid = {
                'n_estimators': [400, 500],
                'max_depth': [3, 4],
                'learning_rate': [0.03, 0.05],
                'subsample': [0.8],
                'scale_pos_weight': [scale_pos_weight]
            }
            
            print(f"🤖 Starting model training with RandomizedSearchCV...")
            print(f"⚙️ Hyperparameter grid: {param_grid}")
            
            model = XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist')
            
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=8,  # Further reduced for stability
                cv=TimeSeriesSplit(n_splits=3),
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,  # Increased verbosity for debugging
                random_state=42
            )
            
            search.fit(X_train_selected, y_train)
            self.model = search.best_estimator_
            
            print(f"✅ Model training completed")
            print(f"🏆 Best parameters: {search.best_params_}")
            print(f"🏆 Best cross-validation score: {search.best_score_:.4f}")
            
            # Find optimal threshold
            print(f"📊 Finding optimal classification threshold...")
            y_pred_proba = self.model.predict_proba(X_test_selected)[:, 1]
            self.best_threshold = self._find_optimal_threshold(y_pred_proba, y_test)
            
            # Calculate final accuracy
            y_pred = (y_pred_proba >= self.best_threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"🎯 Optimal threshold: {self.best_threshold}")
            print(f"📈 Test accuracy: {accuracy:.2%}")
            print(f"📊 Test predictions - UP: {(y_pred == 1).sum()}, DOWN: {(y_pred == 0).sum()}")
            
            # Additional metrics
            from sklearn.metrics import classification_report, confusion_matrix
            print(f"📋 Classification Report:\n{classification_report(y_test, y_pred)}")
            print(f"📋 Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
            
            return accuracy
                
        except Exception as e:
            print(f"❌ Error training model for {symbol}: {str(e)}")
            import traceback
            print(f"🔍 Full error traceback:\n{traceback.format_exc()}")
            return None
    
    def _find_optimal_threshold(self, y_pred_proba, y_test):
        """Find best classification threshold"""
        thresholds = [0.45, 0.48, 0.50, 0.52, 0.55]
        best_threshold, best_acc = 0.5, 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            acc = accuracy_score(y_test, y_pred)
            if acc > best_acc:
                best_acc, best_threshold = acc, threshold
        
        return best_threshold
    
    def predict_next_day(self, symbol):
        """Predict next trading day movement"""
        try:
            if self.model is None:
                print(f"🔄 Training model for {symbol}...")
                accuracy = self.train_model(symbol)
                if accuracy is None:
                    return None
                print(f"✅ Model trained with accuracy: {accuracy:.2%}")
            
            # Get latest data
            df = yf.download(symbol, period='3mo', progress=False)
            if df.empty:
                print(f"❌ No data downloaded for {symbol}")
                return None
            
            df_processed = self.prepare_features(df, symbol)
            if df_processed.empty:
                print(f"❌ Feature preparation failed for {symbol}")
                return None
            
            if self.features is None:
                print(f"❌ No features available for {symbol}")
                return None
            
            # Check if all required features exist
            missing_features = [f for f in self.features if f not in df_processed.columns]
            if missing_features:
                print(f"❌ Missing features for prediction: {missing_features}")
                return None
            
            # Prepare features for prediction
            X = df_processed[self.features]
            X_scaled = self.scaler.transform(X)
            X_selected = self.selector.transform(X_scaled)
            
            # Make prediction
            latest_features = X_selected[-1:].reshape(1, -1)
            next_day_proba = self.model.predict_proba(latest_features)[0, 1]
            next_day_pred = (next_day_proba >= self.best_threshold)
            confidence = next_day_proba if next_day_pred else (1 - next_day_proba)
            
            latest_close = df_processed['Close'].iloc[-1]
            latest_date = df_processed.index[-1]
            
            # Calculate next trading day
            next_day = latest_date + timedelta(days=1)
            while next_day.weekday() >= 5:  # Skip weekends
                next_day += timedelta(days=1)
            
            # Convert all float32 to float64 for JSON serialization
            result = {
                'symbol': symbol,
                'latest_date': latest_date.strftime('%Y-%m-%d'),
                'latest_close': float(round(float(latest_close), 2)),
                'prediction': 'UP' if next_day_pred else 'DOWN',
                'probability': float(round(float(next_day_proba), 4)),
                'confidence': float(round(float(confidence), 4)),
                'next_trading_day': next_day.strftime('%Y-%m-%d'),
                'confidence_level': 'high' if confidence > 0.7 else 'medium' if confidence > 0.6 else 'low'
            }
            
            print(f"✅ Prediction successful for {symbol}: {result}")
            return result
                
        except Exception as e:
            print(f"❌ Prediction error for {symbol}: {e}")
            import traceback
            print(f"🔍 Full error traceback:\n{traceback.format_exc()}")
            return None