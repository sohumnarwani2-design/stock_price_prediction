# Simple backtest runner
from config.settings import Config
from backtest_engine import BacktestEngine

def run_and_print_backtest():
    """Run backtest and print results"""
    print("🚀 STARTING BACKTEST")
    print("=" * 60)
    
    # Initialize
    config = Config()
    engine = BacktestEngine(config, initial_capital=100000)
    
    # Test stocks (using BSE codes from your config)
    test_stocks = ['532540', '500325', '500112']  # TCS, Reliance, SBI
    
    try:
        # Run backtest
        result = engine.execute_backtest(
            bse_codes=test_stocks,
            start_date='2023-01-01',
            end_date='2023-06-30', 
            strategy='moderate'
        )
        
        # Print results
        print("\n📊 BACKTEST RESULTS")
        print("=" * 40)
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Annual Return: {result.annual_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Profitable Trades: {result.profitable_trades}")
        
        # Additional metrics
        print(f"\n📈 ADDITIONAL METRICS")
        print("=" * 40)
        print(f"Average Profit: {result.metrics['avg_profit_percent']:.2f}%")
        print(f"Average Loss: {result.metrics['avg_loss_percent']:.2f}%")
        print(f"Profit Factor: {result.metrics['profit_factor']:.2f}")
        print(f"Average Holding Days: {result.metrics['avg_holding_days']:.1f}")
        print(f"Total P&L: ₹{result.metrics['total_pnl']:,.2f}")
        
        # Sample trades
        if result.trades:
            print(f"\n🎯 SAMPLE TRADES (First 5)")
            print("=" * 40)
            for i, trade in enumerate(result.trades[:5]):
                status = "✅ PROFIT" if trade['profitable'] else "❌ LOSS"
                print(f"{i+1}. {trade['bse_code']} - {trade['entry_date']} to {trade['exit_date']}")
                print(f"   P&L: ₹{trade['pnl']:,.2f} ({trade['pnl_percent']:.2f}%) - {status}")
                print(f"   Signal: {trade['signal']}, Holding: {trade['holding_days']} days")
                print()
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")

# Run it
if __name__ == "__main__":
    run_and_print_backtest()