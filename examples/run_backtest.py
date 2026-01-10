#!/usr/bin/env python3
"""
Example script demonstrating how to use the backtesting framework.

This script shows various ways to run backtests programmatically,
including using different strategies and custom parameters.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest import BacktestEngine, BacktestConfig, run_backtest
from src.strategies import (
    SMACrossoverStrategy,
    RSIMeanReversionStrategy,
    MACDStrategy,
)


def example_basic_backtest():
    """
    Example 1: Basic backtest using convenience function.

    This is the simplest way to run a backtest.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic SMA Crossover Backtest")
    print("=" * 60)

    strategy = SMACrossoverStrategy(fast_period=20, slow_period=50)

    result = run_backtest(
        strategy=strategy,
        ticker="BTC",
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=10000,
        fee_rate=0.001,
    )

    result.print_report()

    # Access individual metrics
    print(f"\nKey Metrics:")
    print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"  Total Trades: {result.metrics.total_trades}")


def example_custom_config():
    """
    Example 2: Backtest with custom configuration.

    This shows how to use BacktestEngine directly with custom settings.
    """
    print("\n" + "=" * 60)
    print("Example 2: RSI Strategy with Custom Config")
    print("=" * 60)

    # Create custom configuration
    config = BacktestConfig(
        initial_capital=50000,
        fee_rate=0.0005,     # 0.05% fee (lower for high-volume traders)
        slippage=0.001,      # 0.1% slippage
        position_size=0.5,   # Use 50% of capital per trade
    )

    # Create engine with config
    engine = BacktestEngine(config)

    # Run with RSI strategy
    strategy = RSIMeanReversionStrategy(
        period=14,
        oversold=25,      # More aggressive oversold level
        overbought=75,    # More aggressive overbought level
    )

    result = engine.run(
        strategy=strategy,
        ticker="ETH",
        start_date="2024-01-01",
        end_date="2024-12-31",
        interval="1d",
    )

    result.print_report()


def example_multiple_strategies():
    """
    Example 3: Compare multiple strategies on the same data.

    This shows how to evaluate different strategies side by side.
    """
    print("\n" + "=" * 60)
    print("Example 3: Strategy Comparison")
    print("=" * 60)

    strategies = [
        SMACrossoverStrategy(fast_period=10, slow_period=30),
        SMACrossoverStrategy(fast_period=20, slow_period=50),
        RSIMeanReversionStrategy(period=14),
        MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
    ]

    config = BacktestConfig(initial_capital=10000, fee_rate=0.001)
    engine = BacktestEngine(config)

    results = []
    for strategy in strategies:
        result = engine.run(
            strategy=strategy,
            ticker="BTC",
            start_date="2024-01-01",
            end_date="2024-12-31",
            generate_charts=False,  # Skip charts for comparison
        )
        results.append(result)

    # Print comparison table
    print("\nStrategy Comparison:")
    print("-" * 90)
    print(f"{'Strategy':<35} {'Return':>10} {'Sharpe':>10} {'Max DD':>10} {'Win Rate':>10} {'Trades':>8}")
    print("-" * 90)

    for result in results:
        m = result.metrics
        print(
            f"{result.strategy_name:<35} "
            f"{m.total_return:>9.1%} "
            f"{m.sharpe_ratio:>10.2f} "
            f"{m.max_drawdown:>9.1%} "
            f"{m.win_rate:>9.1%} "
            f"{m.total_trades:>8}"
        )

    print("-" * 90)


def example_custom_strategy():
    """
    Example 4: Create and test a custom strategy.

    This demonstrates how to implement your own strategy.
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Strategy")
    print("=" * 60)

    from src.strategies.base import Strategy, Signal
    import pandas as pd

    class BollingerBandStrategy(Strategy):
        """
        Bollinger Band mean reversion strategy.

        Buy when price touches lower band, sell when it touches upper band.
        """

        def __init__(self, period: int = 20, num_std: float = 2.0):
            super().__init__(name="Bollinger Bands")
            self.period = period
            self.num_std = num_std

        def generate_signals(self, df: pd.DataFrame) -> pd.Series:
            close = df["close"]
            upper, middle, lower = self.bollinger_bands(close, self.period, self.num_std)

            signals = pd.Series(Signal.HOLD, index=df.index)

            # Buy when close crosses below lower band
            signals[self.crossunder(close, lower)] = Signal.BUY

            # Sell when close crosses above upper band
            signals[self.crossover(close, upper)] = Signal.SELL

            return signals

        def get_params(self) -> dict:
            return {"period": self.period, "num_std": self.num_std}

    # Test the custom strategy
    strategy = BollingerBandStrategy(period=20, num_std=2.0)

    result = run_backtest(
        strategy=strategy,
        ticker="BTC",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    result.print_report()


def example_analyze_trades():
    """
    Example 5: Detailed trade analysis.

    This shows how to access and analyze individual trades.
    """
    print("\n" + "=" * 60)
    print("Example 5: Trade Analysis")
    print("=" * 60)

    strategy = SMACrossoverStrategy(fast_period=20, slow_period=50)

    result = run_backtest(
        strategy=strategy,
        ticker="BTC",
        start_date="2024-01-01",
        end_date="2024-12-31",
        generate_charts=False,
    )

    # Get trades as DataFrame
    trades_df = result.get_trades_df()

    if len(trades_df) > 0:
        print(f"\nTotal trades: {len(trades_df)}")
        print("\nTrade Details:")
        print(trades_df.to_string(index=False))

        # Calculate additional statistics
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]

        print(f"\nWinning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")

        if len(winning_trades) > 0:
            print(f"Average Win: ${winning_trades['pnl'].mean():,.2f}")
            print(f"Largest Win: ${winning_trades['pnl'].max():,.2f}")

        if len(losing_trades) > 0:
            print(f"Average Loss: ${losing_trades['pnl'].mean():,.2f}")
            print(f"Largest Loss: ${losing_trades['pnl'].min():,.2f}")
    else:
        print("No trades were executed.")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("#" + " " * 20 + "BACKTEST EXAMPLES" + " " * 21 + "#")
    print("#" * 60)

    # Run examples
    example_basic_backtest()
    example_custom_config()
    example_multiple_strategies()
    example_custom_strategy()
    example_analyze_trades()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Check the 'output' directory for generated charts.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
