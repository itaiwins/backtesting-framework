"""
Command-line interface for the backtesting framework.

This module provides a CLI for running backtests, listing available
strategies, and fetching historical data.

Usage:
    python -m src.cli backtest --strategy sma_cross --ticker BTC --start 2024-01-01 --end 2024-12-31
    python -m src.cli list-strategies
    python -m src.cli fetch --ticker BTC --start 2024-01-01 --end 2024-12-31
"""

import argparse
import sys
from datetime import datetime

from .backtest import BacktestEngine, BacktestConfig
from .data import DataFetcher
from .strategies import get_strategy, STRATEGY_REGISTRY


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="backtesting-framework",
        description="A modular backtesting framework for crypto trading strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run SMA crossover backtest on BTC
    python -m src.cli backtest --strategy sma_cross --ticker BTC --start 2024-01-01 --end 2024-12-31

    # Run RSI strategy with custom parameters
    python -m src.cli backtest --strategy rsi --ticker ETH --start 2024-06-01 --fee 0.002

    # List available strategies
    python -m src.cli list-strategies

    # Fetch and display historical data
    python -m src.cli fetch --ticker BTC --start 2024-01-01 --end 2024-03-01
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backtest command
    bt_parser = subparsers.add_parser(
        "backtest",
        help="Run a backtest with the specified strategy"
    )
    bt_parser.add_argument(
        "--strategy", "-s",
        type=str,
        required=True,
        help="Strategy name (e.g., sma_cross, rsi, macd)"
    )
    bt_parser.add_argument(
        "--ticker", "-t",
        type=str,
        required=True,
        help="Ticker symbol (e.g., BTC, ETH, SOL)"
    )
    bt_parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    bt_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), defaults to today"
    )
    bt_parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)"
    )
    bt_parser.add_argument(
        "--fee",
        type=float,
        default=0.001,
        help="Fee rate per trade (default: 0.001 = 0.1%%)"
    )
    bt_parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval (default: 1d)"
    )
    bt_parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation"
    )
    bt_parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for charts (default: output)"
    )

    # Strategy-specific parameters
    bt_parser.add_argument(
        "--fast-period",
        type=int,
        default=None,
        help="Fast period for SMA/MACD strategies"
    )
    bt_parser.add_argument(
        "--slow-period",
        type=int,
        default=None,
        help="Slow period for SMA/MACD strategies"
    )
    bt_parser.add_argument(
        "--rsi-period",
        type=int,
        default=14,
        help="RSI period (default: 14)"
    )
    bt_parser.add_argument(
        "--oversold",
        type=float,
        default=30,
        help="RSI oversold threshold (default: 30)"
    )
    bt_parser.add_argument(
        "--overbought",
        type=float,
        default=70,
        help="RSI overbought threshold (default: 70)"
    )

    # List strategies command
    subparsers.add_parser(
        "list-strategies",
        help="List available trading strategies"
    )

    # Fetch data command
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch and display historical data"
    )
    fetch_parser.add_argument(
        "--ticker", "-t",
        type=str,
        required=True,
        help="Ticker symbol"
    )
    fetch_parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    fetch_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )
    fetch_parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval (default: 1d)"
    )

    return parser.parse_args()


def create_strategy(args: argparse.Namespace):
    """Create a strategy instance based on CLI arguments."""
    strategy_class = get_strategy(args.strategy)
    strategy_name = args.strategy.lower()

    # Build kwargs based on strategy type
    kwargs = {}

    if strategy_name in ("sma_cross", "sma"):
        if args.fast_period is not None:
            kwargs["fast_period"] = args.fast_period
        if args.slow_period is not None:
            kwargs["slow_period"] = args.slow_period

    elif strategy_name in ("rsi", "rsi_reversion"):
        kwargs["period"] = args.rsi_period
        kwargs["oversold"] = args.oversold
        kwargs["overbought"] = args.overbought

    elif strategy_name == "macd":
        if args.fast_period is not None:
            kwargs["fast_period"] = args.fast_period
        if args.slow_period is not None:
            kwargs["slow_period"] = args.slow_period

    return strategy_class(**kwargs)


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run backtest command."""
    try:
        # Create strategy
        strategy = create_strategy(args)

        # Create backtest config
        config = BacktestConfig(
            initial_capital=args.capital,
            fee_rate=args.fee,
        )

        # Run backtest
        engine = BacktestEngine(config)
        result = engine.run(
            strategy=strategy,
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
            generate_charts=not args.no_charts,
            output_dir=args.output,
        )

        # Print results
        result.print_report()

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        raise


def cmd_list_strategies(args: argparse.Namespace) -> int:
    """List available strategies."""
    print("\nAvailable Strategies:")
    print("=" * 50)

    # Group by actual strategy class
    shown = set()
    for name, cls in STRATEGY_REGISTRY.items():
        if cls not in shown:
            shown.add(cls)
            aliases = [n for n, c in STRATEGY_REGISTRY.items() if c == cls]
            print(f"\n{cls.__name__}")
            print(f"  Aliases: {', '.join(aliases)}")
            print(f"  {cls.__doc__.split(chr(10))[1].strip() if cls.__doc__ else 'No description'}")

    print("\n" + "=" * 50)
    print("\nUse: python -m src.cli backtest --strategy <name> --ticker <ticker> --start <date>")

    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Fetch and display historical data."""
    try:
        fetcher = DataFetcher()
        data = fetcher.fetch(
            ticker=args.ticker,
            interval=args.interval,
            start_date=args.start,
            end_date=args.end,
        )

        print(f"\n{data}")
        print("\nFirst 10 rows:")
        print(data.df.head(10))
        print("\nLast 10 rows:")
        print(data.df.tail(10))
        print("\nSummary statistics:")
        print(data.df.describe())

        return 0

    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for CLI."""
    args = parse_args()

    if args.command == "backtest":
        return cmd_backtest(args)
    elif args.command == "list-strategies":
        return cmd_list_strategies(args)
    elif args.command == "fetch":
        return cmd_fetch(args)
    else:
        print("Please specify a command. Use --help for available commands.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
