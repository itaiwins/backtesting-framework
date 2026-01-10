"""
Main backtesting engine module.

This module contains the BacktestEngine class which orchestrates the entire
backtesting process: loading data, generating signals, simulating trades,
and calculating performance metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from .data import DataFetcher, OHLCVData
from .metrics import (
    PerformanceMetrics,
    calculate_metrics,
    format_metrics_report,
)
from .visualization import generate_backtest_charts
from .strategies.base import Strategy, Signal


@dataclass
class BacktestConfig:
    """
    Configuration for a backtest run.

    Attributes:
        initial_capital: Starting portfolio value in dollars.
        fee_rate: Trading fee as a decimal (e.g., 0.001 = 0.1%).
        slippage: Estimated slippage as a decimal (e.g., 0.0005 = 0.05%).
        position_size: Fraction of capital to use per trade (1.0 = 100%).
        allow_short: Whether to allow short selling (not yet implemented).
    """

    initial_capital: float = 10000.0
    fee_rate: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    position_size: float = 1.0  # Use 100% of available capital
    allow_short: bool = False  # Long-only for now


@dataclass
class Trade:
    """Record of a single completed trade."""

    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    fees_paid: float


@dataclass
class BacktestResult:
    """
    Complete results from a backtest run.

    Contains all data needed for analysis and visualization.
    """

    strategy_name: str
    ticker: str
    start_date: str
    end_date: str
    config: BacktestConfig
    metrics: PerformanceMetrics
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trades: list[Trade]
    signals: pd.Series
    positions: pd.Series

    def get_trades_df(self) -> pd.DataFrame:
        """Convert trades list to DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "fees_paid": t.fees_paid,
            }
            for t in self.trades
        ])

    def print_report(self) -> None:
        """Print formatted performance report to console."""
        report = format_metrics_report(
            self.metrics,
            self.strategy_name,
            self.ticker,
            self.start_date,
            self.end_date
        )
        print(report)


class BacktestEngine:
    """
    Main backtesting engine for running strategy simulations.

    The engine follows these steps:
    1. Load historical OHLCV data
    2. Generate signals from the strategy
    3. Convert signals to positions
    4. Simulate trades with fees and slippage
    5. Calculate equity curve and performance metrics
    6. Generate visualizations

    Example:
        >>> from src.backtest import BacktestEngine, BacktestConfig
        >>> from src.strategies import SMACrossoverStrategy
        >>>
        >>> engine = BacktestEngine()
        >>> strategy = SMACrossoverStrategy(fast_period=20, slow_period=50)
        >>> result = engine.run(
        ...     strategy=strategy,
        ...     ticker="BTC",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31"
        ... )
        >>> result.print_report()
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize the backtesting engine.

        Args:
            config: Backtest configuration (uses defaults if not provided).
        """
        self.config = config or BacktestConfig()
        self.data_fetcher = DataFetcher()

    def run(
        self,
        strategy: Strategy,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
        generate_charts: bool = True,
        output_dir: str = "output"
    ) -> BacktestResult:
        """
        Run a complete backtest.

        Args:
            strategy: Strategy instance to test.
            ticker: Ticker symbol (e.g., "BTC", "ETH").
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format (defaults to today).
            interval: Data interval (e.g., "1d", "4h", "1h").
            generate_charts: Whether to generate and save charts.
            output_dir: Directory for saving charts.

        Returns:
            BacktestResult containing all backtest data and metrics.
        """
        # Fetch data
        print(f"Fetching {ticker} data from {start_date} to {end_date}...")
        ohlcv = self.data_fetcher.fetch(ticker, interval, start_date, end_date)
        df = ohlcv.df

        print(f"Loaded {len(df)} candles")

        # Generate signals
        print(f"Generating signals using {strategy.name}...")
        signals = strategy.generate_signals(df)

        # Convert signals to positions
        positions = self._signals_to_positions(signals)

        # Simulate trades and build equity curve
        print("Simulating trades...")
        equity_curve, trades = self._simulate_trades(df, positions)

        # Calculate buy & hold benchmark
        benchmark_curve = self._calculate_benchmark(df)

        # Calculate buy & hold return
        buy_hold_return = (
            (df["close"].iloc[-1] - df["close"].iloc[0]) /
            df["close"].iloc[0]
        )

        # Calculate metrics
        trade_returns = [t.pnl_pct for t in trades]
        metrics = calculate_metrics(
            equity_curve=equity_curve,
            trade_returns=trade_returns,
            initial_capital=self.config.initial_capital,
            buy_hold_return=buy_hold_return,
            periods_per_year=self._get_periods_per_year(interval)
        )

        # Create result object
        result = BacktestResult(
            strategy_name=str(strategy),
            ticker=ohlcv.symbol,
            start_date=start_date,
            end_date=end_date or datetime.now().strftime("%Y-%m-%d"),
            config=self.config,
            metrics=metrics,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            trades=trades,
            signals=signals,
            positions=positions,
        )

        # Generate charts
        if generate_charts:
            print("Generating charts...")
            trades_for_chart = self._prepare_trades_for_chart(trades)
            metrics_dict = {
                "total_return": metrics.total_return,
                "buy_hold_return": metrics.buy_hold_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "total_trades": metrics.total_trades,
                "profit_factor": metrics.profit_factor,
            }
            charts = generate_backtest_charts(
                equity_curve=equity_curve,
                benchmark=benchmark_curve,
                trades=trades_for_chart,
                strategy_name=strategy.name,
                ticker=ohlcv.symbol,
                metrics=metrics_dict,
                output_dir=output_dir,
            )
            print(f"Charts saved to: {output_dir}/")

        return result

    def _signals_to_positions(self, signals: pd.Series) -> pd.Series:
        """
        Convert trading signals to position states.

        This handles the logic of entering and exiting positions:
        - BUY signal: Enter long position (position = 1)
        - SELL signal: Exit position (position = 0)
        - HOLD signal: Maintain current position

        Args:
            signals: Series of Signal values.

        Returns:
            Series of position states (0 = no position, 1 = long).
        """
        positions = pd.Series(0, index=signals.index)
        in_position = False

        for i, (idx, signal) in enumerate(signals.items()):
            if signal == Signal.BUY and not in_position:
                in_position = True
            elif signal == Signal.SELL and in_position:
                in_position = False

            positions.iloc[i] = 1 if in_position else 0

        return positions

    def _simulate_trades(
        self,
        df: pd.DataFrame,
        positions: pd.Series
    ) -> tuple[pd.Series, list[Trade]]:
        """
        Simulate trades and calculate equity curve.

        Args:
            df: OHLCV DataFrame.
            positions: Position series (0 or 1).

        Returns:
            Tuple of (equity_curve, list_of_trades).
        """
        initial_capital = self.config.initial_capital
        fee_rate = self.config.fee_rate
        slippage = self.config.slippage
        position_size = self.config.position_size

        # Track state
        cash = initial_capital
        shares = 0.0
        entry_price = 0.0
        entry_date = None
        trades: list[Trade] = []

        # Build equity curve
        equity = []

        for i in range(len(df)):
            idx = df.index[i]
            close = df["close"].iloc[i]
            pos = positions.iloc[i]
            prev_pos = positions.iloc[i - 1] if i > 0 else 0

            # Check for position changes
            if pos == 1 and prev_pos == 0:
                # Enter position
                entry_price = close * (1 + slippage)  # Pay slippage on entry
                available_capital = cash * position_size
                fee = available_capital * fee_rate
                shares = (available_capital - fee) / entry_price
                cash = cash - (shares * entry_price) - fee
                entry_date = idx

            elif pos == 0 and prev_pos == 1:
                # Exit position
                exit_price = close * (1 - slippage)  # Pay slippage on exit
                proceeds = shares * exit_price
                fee = proceeds * fee_rate
                pnl = proceeds - fee - (shares * entry_price)
                pnl_pct = (exit_price / entry_price) - 1 - (2 * fee_rate)

                trades.append(Trade(
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=idx,
                    exit_price=exit_price,
                    shares=shares,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    fees_paid=fee * 2,  # Entry + exit fees
                ))

                cash = cash + proceeds - fee
                shares = 0.0

            # Calculate current equity
            if shares > 0:
                current_equity = cash + (shares * close)
            else:
                current_equity = cash

            equity.append(current_equity)

        equity_curve = pd.Series(equity, index=df.index)

        return equity_curve, trades

    def _calculate_benchmark(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate buy & hold benchmark equity curve.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series of buy & hold equity values.
        """
        initial_price = df["close"].iloc[0]
        shares = self.config.initial_capital / initial_price
        benchmark = df["close"] * shares

        return benchmark

    def _get_periods_per_year(self, interval: str) -> int:
        """
        Get the number of trading periods per year for an interval.

        Args:
            interval: Data interval string.

        Returns:
            Approximate number of periods per year.
        """
        interval_periods = {
            "1m": 525600,    # 365 * 24 * 60
            "5m": 105120,    # 365 * 24 * 12
            "15m": 35040,    # 365 * 24 * 4
            "30m": 17520,    # 365 * 24 * 2
            "1h": 8760,      # 365 * 24
            "4h": 2190,      # 365 * 6
            "1d": 365,
            "1w": 52,
            "1M": 12,
        }
        return interval_periods.get(interval, 365)

    def _prepare_trades_for_chart(self, trades: list[Trade]) -> pd.DataFrame:
        """
        Prepare trades data for chart visualization.

        Args:
            trades: List of Trade objects.

        Returns:
            DataFrame with trade entries and exits.
        """
        if not trades:
            return pd.DataFrame(columns=["date", "type", "price"])

        records = []
        for t in trades:
            records.append({
                "date": t.entry_date,
                "type": "entry",
                "price": t.entry_price,
            })
            records.append({
                "date": t.exit_date,
                "type": "exit",
                "price": t.exit_price,
            })

        return pd.DataFrame(records)


def run_backtest(
    strategy: Strategy,
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    generate_charts: bool = True,
    output_dir: str = "output"
) -> BacktestResult:
    """
    Convenience function to run a backtest with common parameters.

    This is a simpler interface than creating a BacktestEngine directly.

    Args:
        strategy: Strategy instance to test.
        ticker: Ticker symbol.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        initial_capital: Starting capital.
        fee_rate: Fee per trade as decimal.
        generate_charts: Whether to generate charts.
        output_dir: Output directory for charts.

    Returns:
        BacktestResult with all metrics and data.

    Example:
        >>> from src.strategies import SMACrossoverStrategy
        >>> from src.backtest import run_backtest
        >>>
        >>> result = run_backtest(
        ...     strategy=SMACrossoverStrategy(20, 50),
        ...     ticker="BTC",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31"
        ... )
        >>> result.print_report()
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
    )
    engine = BacktestEngine(config)

    return engine.run(
        strategy=strategy,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        generate_charts=generate_charts,
        output_dir=output_dir,
    )
