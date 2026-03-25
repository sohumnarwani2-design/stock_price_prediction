# backtest_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestResult:
    """Container for backtest results"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    trades: List[Dict]
    equity_curve: pd.Series
    metrics: Dict

class BacktestEngine:
    def __init__(self, config, initial_capital: float = 100000):
        self.config = config
        self.initial_capital = initial_capital
        self.results = {}
        
        # Initialize services
        from services.sent_an import OptimizedSentimentAnalyzer
        from services.price_pred import OptimizedPricePredictor
        from services.signal_fuser import SignalFuser
        from utils.helpers import get_bse_client
        
        self.sentiment_analyzer = OptimizedSentimentAnalyzer(config)
        self.price_predictor = OptimizedPricePredictor(config)
        self.signal_fuser = SignalFuser(config)
        self.bse_client = get_bse_client(config)
        
        # Backtest parameters
        self.position_size = 0.2  #20% of account value per trade
        self.stop_loss = 0.05     #5% stop loss
        self.take_profit = 0.025  #2.5% take profit
        self.holding_period = 10  #max holding period of 10 days
        
        # Cache for trained models and data
        self.trained_models = {}
        self.historical_data_cache = {}
        self.sentiment_cache = {}

    def robust_yfinance_download(self, symbol: str, start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:
        """Robust yfinance download with retries and error handling"""
        for attempt in range(max_retries):
            try:
                print(f"📥 Download attempt {attempt+1} for {symbol}...")
                df = yf.download(symbol, start=start_date, end=end_date, progress=False, timeout=30)
                
                if df.empty:
                    print(f"❌ Empty data for {symbol} on attempt {attempt+1}")
                    continue
                
                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    print(f"🔍 MultiIndex detected for {symbol}, extracting data...")
                    if symbol in df.columns.get_level_values(1):
                        df_clean = df.xs(symbol, axis=1, level=1)
                    else:
                        # Try to use the first available symbol
                        available_symbols = df.columns.get_level_values(1).unique()
                        if len(available_symbols) > 0:
                            first_symbol = available_symbols[0]
                            df_clean = df.xs(first_symbol, axis=1, level=1)
                            print(f"⚠️ Using {first_symbol} instead of {symbol}")
                        else:
                            print(f"❌ No symbols found in MultiIndex for {symbol}")
                            continue
                else:
                    df_clean = df.copy()
                
                # Ensure we have required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in df_clean.columns]
                if missing_cols:
                    print(f"❌ Missing columns {missing_cols} for {symbol}")
                    continue
                
                print(f"✅ Successfully downloaded data for {symbol}, shape: {df_clean.shape}")
                return df_clean[required_cols]
                
            except Exception as e:
                print(f"❌ Download failed for {symbol} on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    print("🔄 Retrying...")
                    continue
                else:
                    print(f"💥 All download attempts failed for {symbol}")
                    return pd.DataFrame()

    def prepare_models_and_data(self, bse_codes: List[str], start_date: str, end_date: str):
        """Pre-train models and load data once at the beginning"""
        print("🔄 Preparing models and data...")
        
        successful_stocks = []
        
        for bse_code in bse_codes:
            symbol = self.config.STOCK_MAPPING[bse_code][0]
            
            # Load historical data with robust download
            print(f"📥 Loading data for {symbol}...")
            data = self.robust_yfinance_download(symbol, start_date, end_date)
            
            if not data.empty:
                self.historical_data_cache[bse_code] = data
                successful_stocks.append(bse_code)
                
                # Train price prediction model once
                print(f"🤖 Training model for {symbol}...")
                try:
                    accuracy = self.price_predictor.train_model(symbol)
                    if accuracy is not None:
                        self.trained_models[symbol] = {
                            'model': self.price_predictor.model,
                            'scaler': self.price_predictor.scaler,
                            'selector': self.price_predictor.selector,
                            'features': self.price_predictor.features,
                            'threshold': self.price_predictor.best_threshold
                        }
                        print(f"✅ Model trained for {symbol} with accuracy: {accuracy:.2%}")
                    else:
                        print(f"❌ Model training failed for {symbol}")
                except Exception as e:
                    print(f"❌ Model training error for {symbol}: {e}")
            
            # Pre-load sentiment data
            self.preload_sentiment_data(bse_code, start_date, end_date)
        
        print(f"🎯 Successfully prepared {len(successful_stocks)} stocks: {successful_stocks}")
        return successful_stocks

    def preload_sentiment_data(self, bse_code: str, start_date: str, end_date: str):
        """Pre-load sentiment data for the entire period"""
        try:
            # Get all announcements for the entire period
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days = (end_dt - start_dt).days + 30  # Buffer for lookback
            
            announcements = self.bse_client.get_corporate_announcements(bse_code, days=days)
            self.sentiment_cache[bse_code] = announcements
            print(f"✅ Pre-loaded {len(announcements)} announcements for {bse_code}")
            
        except Exception as e:
            print(f"❌ Failed to pre-load sentiment data for {bse_code}: {e}")
            self.sentiment_cache[bse_code] = []

    def get_sentiment_for_date(self, bse_code: str, date: datetime) -> Dict:
        """Get sentiment for a specific date using cached data"""
        try:
            announcements = self.sentiment_cache.get(bse_code, [])
            
            # Filter announcements up to the current date
            valid_announcements = []
            for ann in announcements:
                try:
                    ann_date = datetime.strptime(ann['date'], '%d-%b-%Y')
                    if ann_date <= date:
                        valid_announcements.append(ann)
                except:
                    continue
            
            if valid_announcements:
                return self.sentiment_analyzer.analyze_news_sentiment(valid_announcements)
            else:
                return self.sentiment_analyzer._get_neutral_response()
                
        except Exception as e:
            print(f"❌ Sentiment analysis failed for {bse_code} on {date}: {e}")
            return self.sentiment_analyzer._get_neutral_response()

    def get_price_prediction_for_date(self, bse_code: str, date: datetime) -> Optional[Dict]:
        """Get price prediction for a specific date using pre-trained model"""
        try:
            symbol = self.config.STOCK_MAPPING[bse_code][0]
            historical_data = self.historical_data_cache.get(bse_code)
            
            if historical_data is None or historical_data.empty:
                return None
            
            # Use data up to the prediction date
            data_up_to_date = historical_data[historical_data.index <= date].copy()
            
            if len(data_up_to_date) < 30:  # Need sufficient data
                return None
            
            # Use the pre-trained model to predict
            model_info = self.trained_models.get(symbol)
            if not model_info:
                return None
            
            # Prepare features for the current date
            df_processed = self.price_predictor.prepare_features(data_up_to_date, symbol)
            if df_processed.empty:
                return None
            
            # Get the latest features
            features = model_info['features']
            if features is None:
                return None
            
            # Check if all required features exist
            missing_features = [f for f in features if f not in df_processed.columns]
            if missing_features:
                return None
            
            # Prepare features for prediction
            X = df_processed[features].iloc[-1:].values  # Only the latest row
            X_scaled = model_info['scaler'].transform(X)
            X_selected = model_info['selector'].transform(X_scaled)
            
            # Make prediction
            next_day_proba = model_info['model'].predict_proba(X_selected)[0, 1]
            next_day_pred = (next_day_proba >= model_info['threshold'])
            confidence = next_day_proba if next_day_pred else (1 - next_day_proba)
            
            latest_close = df_processed['Close'].iloc[-1]
            
            # Calculate next trading day
            next_day = date + timedelta(days=1)
            while next_day.weekday() >= 5:  # Skip weekends
                next_day += timedelta(days=1)
            
            return {
                'symbol': symbol,
                'latest_date': date.strftime('%Y-%m-%d'),
                'latest_close': float(latest_close),
                'prediction': 'UP' if next_day_pred else 'DOWN',
                'probability': float(next_day_proba),
                'confidence': float(confidence),
                'next_trading_day': next_day.strftime('%Y-%m-%d'),
                'confidence_level': 'high' if confidence > 0.7 else 'medium' if confidence > 0.6 else 'low'
            }
                
        except Exception as e:
            print(f"❌ Price prediction failed for {bse_code} on {date}: {e}")
            return None

    def generate_trading_signal(self, bse_code: str, date: datetime) -> Optional[Dict]:
        """Generate trading signal for a specific date"""
        try:
            # Get sentiment analysis
            sentiment_result = self.get_sentiment_for_date(bse_code, date)
            
            # Get price prediction
            price_result = self.get_price_prediction_for_date(bse_code, date)
            
            if sentiment_result and price_result:
                # Fuse signals
                fused_signal = self.signal_fuser.fuse_signals(sentiment_result, price_result)
                
                signal_data = {
                    'date': date.strftime('%Y-%m-%d'),
                    'bse_code': bse_code,
                    'symbol': self.config.STOCK_MAPPING[bse_code][0],
                    'sentiment': sentiment_result,
                    'price_prediction': price_result,
                    'fused_signal': fused_signal,
                    'current_price': price_result['latest_close'],
                    'signal_strength': fused_signal['signal_strength'],
                    'final_signal': fused_signal['final_signal']
                }
                
                return signal_data
            else:
                return None
                
        except Exception as e:
            print(f"❌ Signal generation failed for {bse_code} on {date}: {e}")
            return None

    def execute_backtest(self, bse_codes: List[str], start_date: str, end_date: str, 
                        strategy: str = "moderate") -> BacktestResult:
        """Execute backtest for multiple stocks"""
        print("🚀 Starting backtest...")
        
        # Pre-load all data and train models
        successful_stocks = self.prepare_models_and_data(bse_codes, start_date, end_date)
        
        if not successful_stocks:
            print("❌ No stocks with valid data for backtesting")
            return self._get_empty_result()
        
        all_trades = []
        cash = self.initial_capital
        positions = {}
        equity_curve = []
        dates = []
        
        # Generate trading dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        trading_dates = self._generate_trading_dates(start_dt, end_dt)
        
        print(f"📅 Backtesting {len(trading_dates)} trading days for {len(successful_stocks)} stocks...")
        
        for i, current_date in enumerate(trading_dates):
            if i % 10 == 0:  # Progress update
                print(f"📊 Processing date {i+1}/{len(trading_dates)}: {current_date.strftime('%Y-%m-%d')}")
            
            # Update portfolio value based on current positions
            current_portfolio_value = cash
            positions_to_remove = []
            
            for bse_code, position in positions.items():
                historical_data = self.historical_data_cache.get(bse_code)
                if historical_data is None or current_date not in historical_data.index:
                    continue
                
                # Get current price - ensure it's a scalar value
                try:
                    current_price = float(historical_data.loc[current_date, 'Close'])
                except:
                    continue
                
                position_value = position['shares'] * current_price
                current_portfolio_value += position_value
                
                # Update position current value
                positions[bse_code]['current_price'] = current_price
                positions[bse_code]['current_value'] = position_value
                
                # Check exit conditions
                if self._should_exit_position(position, current_date):
                    # Close position
                    trade = self._close_position(bse_code, position, current_date)
                    if trade and self._is_valid_trade(trade):
                        cash += trade['exit_value']
                        all_trades.append(trade)
                        positions_to_remove.append(bse_code)
            
            # Remove closed positions
            for bse_code in positions_to_remove:
                del positions[bse_code]
            
            equity_curve.append(current_portfolio_value)
            dates.append(current_date)
            
            # Generate new signals and enter positions (limit to 1 new position per day)
            if len(positions) < len(successful_stocks) and cash > 1000:
                for bse_code in successful_stocks:
                    if bse_code not in positions:
                        signal = self.generate_trading_signal(bse_code, current_date)
                        
                        if signal and self._should_enter_trade(signal, strategy):
                            trade_size = min(cash * self.position_size, cash * 0.5)
                            if trade_size > 1000:
                                trade = self._enter_position(bse_code, signal, trade_size, current_date)
                                if trade:
                                    cash -= trade['entry_value']
                                    positions[bse_code] = {
                                        'entry_date': current_date,
                                        'entry_price': trade['entry_price'],
                                        'shares': trade['shares'],
                                        'signal': signal,
                                        'stop_loss': trade['stop_loss'],
                                        'take_profit': trade['take_profit'],
                                        'current_price': trade['entry_price']
                                    }
                                    # Don't append entry trades to all_trades - only closed trades
                                    break  # Only enter one new position per day
        
        # Close all remaining positions at end
        for bse_code in list(positions.keys()):
            trade = self._close_position(bse_code, positions[bse_code], end_dt)
            if trade and self._is_valid_trade(trade):
                cash += trade['exit_value']
                all_trades.append(trade)
        
        print("✅ Backtest completed!")
        return self._calculate_performance_metrics(all_trades, equity_curve, dates, self.initial_capital)

    def _is_valid_trade(self, trade: Dict) -> bool:
        """Check if trade has all required fields"""
        required_fields = ['pnl', 'entry_price', 'exit_price', 'shares']
        return all(field in trade for field in required_fields)

    def _generate_trading_dates(self, start_dt: datetime, end_dt: datetime) -> List[datetime]:
        """Generate list of trading dates (weekdays only)"""
        dates = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            if current_dt.weekday() < 5:  # Monday to Friday
                dates.append(current_dt)
            current_dt += timedelta(days=1)
        
        return dates

    def _should_enter_trade(self, signal: Dict, strategy: str) -> bool:
        """Determine if we should enter a trade based on signal and strategy"""
        final_signal = signal['fused_signal']['final_signal']
        strength = signal['fused_signal']['signal_strength']
        
        if strategy == "conservative":
            return final_signal in ["STRONG_BUY"] and strength == "strong"
        elif strategy == "moderate":
            return final_signal in ["STRONG_BUY", "BUY"] and strength in ["strong", "medium"]
        else:  # aggressive
            return final_signal in ["STRONG_BUY", "BUY"]

    def _should_exit_position(self, position: Dict, current_date: datetime) -> bool:
        """Check if position should be exited - FIXED to handle scalar values"""
        try:
            entry_price = float(position['entry_price'])
            current_price = float(position['current_price'])
            holding_days = (current_date - position['entry_date']).days
            
            # Check holding period
            if holding_days >= self.holding_period:
                return True
            
            # Check stop loss - ensure we're comparing floats
            stop_loss_price = float(position['stop_loss'])
            if current_price <= stop_loss_price:
                return True
            
            # Check take profit
            take_profit_price = float(position['take_profit'])
            if current_price >= take_profit_price:
                return True
            
            return False
        except (KeyError, TypeError, ValueError) as e:
            print(f"❌ Error in exit position check: {e}")
            return True  # Exit on error

    def _enter_position(self, bse_code: str, signal: Dict, trade_size: float, entry_date: datetime) -> Optional[Dict]:
        """Enter a new position"""
        try:
            entry_price = float(signal['current_price'])
            shares = int(trade_size / entry_price)
            
            if shares == 0:
                return None
            
            actual_trade_size = shares * entry_price
            
            return {
                'bse_code': bse_code,
                'symbol': signal['symbol'],
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'entry_price': entry_price,
                'shares': shares,
                'position_type': 'LONG',
                'signal': signal['fused_signal']['final_signal'],
                'stop_loss': entry_price * (1 - self.stop_loss),
                'take_profit': entry_price * (1 + self.take_profit),
                'entry_value': actual_trade_size
            }
        except Exception as e:
            print(f"❌ Error entering position: {e}")
            return None

    def _close_position(self, bse_code: str, position: Dict, exit_date: datetime) -> Optional[Dict]:
        """Close an existing position - FIXED to ensure all required fields"""
        try:
            exit_price = float(position['current_price'])
            shares = position['shares']
            entry_price = position['entry_price']
            
            exit_value = shares * exit_price
            entry_value = shares * entry_price
            pnl = exit_value - entry_value
            pnl_percent = (pnl / entry_value) * 100 if entry_value > 0 else 0
            
            # Get signal information safely
            signal_data = position.get('signal', {})
            if isinstance(signal_data, dict):
                final_signal = signal_data.get('fused_signal', {}).get('final_signal', 'UNKNOWN')
            else:
                final_signal = str(signal_data)
            
            trade = {
                'bse_code': bse_code,
                'symbol': self.config.STOCK_MAPPING.get(bse_code, ['Unknown'])[0],
                'entry_date': position['entry_date'].strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'entry_value': entry_value,
                'exit_value': exit_value,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'holding_days': (exit_date - position['entry_date']).days,
                'profitable': pnl > 0,
                'signal': final_signal,
                'position_type': 'LONG'
            }
            
            return trade
        except Exception as e:
            print(f"❌ Error closing position: {e}")
            return None

    def _calculate_performance_metrics(self, trades: List[Dict], equity_curve: List[float], 
                                     dates: List[datetime], initial_capital: float) -> BacktestResult:
        """Calculate comprehensive performance metrics - FIXED to handle missing pnl"""
        if not trades or not equity_curve:
            return self._get_empty_result()
        
        # Filter only valid trades with pnl
        valid_trades = [t for t in trades if self._is_valid_trade(t)]
        
        if not valid_trades:
            return self._get_empty_result()
        
        # Basic metrics
        total_trades = len(valid_trades)
        profitable_trades = len([t for t in valid_trades if t['pnl'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Returns
        total_return = (equity_curve[-1] - initial_capital) / initial_capital
        days_period = (dates[-1] - dates[0]).days
        annual_return = (1 + total_return) ** (365 / days_period) - 1 if days_period > 0 else 0
        
        # Sharpe ratio
        equity_series = pd.Series(equity_curve, index=dates)
        returns_series = equity_series.pct_change().dropna()
        if len(returns_series) > 1 and returns_series.std() > 0:
            sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Additional metrics
        profitable_pnls = [t['pnl_percent'] for t in valid_trades if t['pnl'] > 0]
        loss_pnls = [t['pnl_percent'] for t in valid_trades if t['pnl'] <= 0]
        
        avg_profit = np.mean(profitable_pnls) if profitable_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0
        
        total_profit = sum([t['pnl'] for t in valid_trades if t['pnl'] > 0])
        total_loss = abs(sum([t['pnl'] for t in valid_trades if t['pnl'] < 0]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        metrics = {
            'avg_profit_percent': avg_profit,
            'avg_loss_percent': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_days': np.mean([t['holding_days'] for t in valid_trades]) if valid_trades else 0,
            'total_pnl': sum(t['pnl'] for t in valid_trades),
            'final_portfolio_value': equity_curve[-1],
            'valid_trades': len(valid_trades),
            'invalid_trades': len(trades) - len(valid_trades)
        }
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            trades=valid_trades,  # Only return valid trades
            equity_curve=equity_series,
            metrics=metrics
        )

    def _get_empty_result(self):
        """Return empty result when no trades occur"""
        return BacktestResult(
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            profitable_trades=0,
            trades=[],
            equity_curve=pd.Series(),
            metrics={'final_portfolio_value': self.initial_capital, 'total_pnl': 0}
        )

# Test function
def test_backtest_engine():
    """Test the backtest engine"""
    print("🧪 Testing Backtest Engine")
    print("=" * 60)
    
    from config.settings import Config
    config = Config()
    
    # Use only stocks that are likely to work
    test_stocks = ['532540', '500325', '500209', '500180', '500696', '500875', '532555']  #TCS, Reliance, Infosys, HDFC Bank, ICICI Bank, NTPC
    
    engine = BacktestEngine(config, initial_capital=100000)
    
    try:
        print("🚀 Running backtest...")
        result = engine.execute_backtest(
            bse_codes=test_stocks,
            start_date='2023-01-01',
            end_date='2023-12-31',  
            strategy='moderate'
        )
        
        print("\n📊 BACKTEST RESULTS")
        print("=" * 40)
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Annual Return: {result.annual_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Profitable Trades: {result.profitable_trades}")
        print(f"Final Portfolio Value: ₹{result.metrics.get('final_portfolio_value', 0):,.2f}")
        print(f"Total P&L: ₹{result.metrics.get('total_pnl', 0):,.2f}")
        
        if result.trades:
            print(f"\n🎯 TRADES EXECUTED ({len(result.trades)} valid trades)")
            print("=" * 40)
            for i, trade in enumerate(result.trades):
                status = "✅ PROFIT" if trade['profitable'] else "❌ LOSS"
                print(f"{i+1}. {trade['bse_code']} - {trade['entry_date']} to {trade['exit_date']}")
                print(f"   P&L: ₹{trade['pnl']:,.2f} ({trade['pnl_percent']:.2f}%) - {status}")
                print(f"   Signal: {trade['signal']}, Holding: {trade['holding_days']} days")
                print()
        else:
            print("📭 No valid trades were executed during the backtest period")
            print("💡 This could be because:")
            print("   - No strong trading signals were generated")
            print("   - Price predictions didn't meet confidence thresholds")
            print("   - Market conditions didn't trigger entry criteria")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")

if __name__ == "__main__":
    test_backtest_engine()