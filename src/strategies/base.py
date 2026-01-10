"""
Base Strategy class for implementing trading strategies.

This module provides the abstract base class that all trading strategies
must inherit from. It defines the interface for signal generation and
provides utilities for common technical indicators.

To create a custom strategy:
1. Inherit from the Strategy class
2. Implement the generate_signals() method
3. Optionally override get_params() for parameter reporting

Example:
    ```python
    from src.strategies.base import Strategy, Signal
    import pandas as pd

    class MyStrategy(Strategy):
        def __init__(self, lookback: int = 20):
            super().__init__(name="My Custom Strategy")
            self.lookback = lookback

        def generate_signals(self, df: pd.DataFrame) -> pd.Series:
            # Your signal logic here
            # Return 1 for buy, -1 for sell, 0 for hold
            signals = pd.Series(Signal.HOLD, index=df.index)

            # Example: Buy when close > SMA
            sma = df['close'].rolling(self.lookback).mean()
            signals[df['close'] > sma] = Signal.BUY
            signals[df['close'] < sma] = Signal.SELL

            return signals

        def get_params(self) -> dict:
            return {"lookback": self.lookback}
    ```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd


class Signal(IntEnum):
    """
    Trading signal values.

    Attributes:
        SELL: Exit long position or enter short (-1)
        HOLD: No action (0)
        BUY: Enter long position (1)
    """

    SELL = -1
    HOLD = 0
    BUY = 1


@dataclass
class TradeInfo:
    """Information about a single trade."""

    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    holding_period: int  # Number of bars


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    All custom strategies must inherit from this class and implement
    the generate_signals() method.

    Attributes:
        name: Human-readable name of the strategy.

    The signal generation follows these conventions:
        - Signal.BUY (1): Enter a long position
        - Signal.SELL (-1): Exit the long position
        - Signal.HOLD (0): Maintain current position

    Note:
        This framework currently supports long-only strategies.
        Short selling can be added by extending the BacktestEngine.
    """

    def __init__(self, name: str = "Unnamed Strategy"):
        """
        Initialize the strategy.

        Args:
            name: Human-readable name for the strategy.
        """
        self.name = name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on price data.

        This method must be implemented by all strategy subclasses.
        It receives OHLCV data and should return a Series of signals.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                indexed by datetime.

        Returns:
            pd.Series: Series of Signal values (BUY=1, SELL=-1, HOLD=0)
                      indexed by datetime, same length as input df.

        Note:
            - The returned Series MUST have the same index as the input DataFrame.
            - Use Signal.BUY, Signal.SELL, Signal.HOLD for clarity.
            - Handle NaN values from rolling calculations appropriately.
        """
        pass

    def get_params(self) -> dict[str, Any]:
        """
        Return strategy parameters for reporting.

        Override this method to include strategy-specific parameters
        in backtest reports.

        Returns:
            Dictionary of parameter names and values.
        """
        return {}

    def __repr__(self) -> str:
        params = self.get_params()
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            return f"{self.name}({param_str})"
        return self.name

    # ==================== Technical Indicator Helpers ====================
    # These static methods provide common indicators that strategies can use.

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.

        Args:
            series: Price series (typically close prices).
            period: Lookback period.

        Returns:
            SMA values.
        """
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            series: Price series (typically close prices).
            period: Lookback period (used to calculate span).

        Returns:
            EMA values.
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        RSI measures the speed and magnitude of price changes.
        Values range from 0 to 100:
        - Above 70: Potentially overbought
        - Below 30: Potentially oversold

        Args:
            series: Price series (typically close prices).
            period: Lookback period (default: 14).

        Returns:
            RSI values (0-100).
        """
        delta = series.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD is a trend-following momentum indicator that shows the
        relationship between two EMAs of the price.

        Args:
            series: Price series (typically close prices).
            fast_period: Fast EMA period (default: 12).
            slow_period: Slow EMA period (default: 26).
            signal_period: Signal line EMA period (default: 9).

        Returns:
            Tuple of (macd_line, signal_line, histogram):
            - macd_line: Difference between fast and slow EMAs
            - signal_line: EMA of the MACD line
            - histogram: Difference between MACD and signal line
        """
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Bollinger Bands consist of a middle band (SMA) and two outer bands
        at a specified number of standard deviations.

        Args:
            series: Price series (typically close prices).
            period: Lookback period for SMA and std (default: 20).
            num_std: Number of standard deviations for bands (default: 2.0).

        Returns:
            Tuple of (upper_band, middle_band, lower_band).
        """
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range.

        ATR measures market volatility by decomposing the entire range
        of an asset price for a period.

        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            period: Lookback period (default: 14).

        Returns:
            ATR values.
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        Detect when series1 crosses above series2.

        Args:
            series1: First series (e.g., fast MA).
            series2: Second series (e.g., slow MA).

        Returns:
            Boolean series that is True when series1 crosses above series2.
        """
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

    @staticmethod
    def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        Detect when series1 crosses below series2.

        Args:
            series1: First series (e.g., fast MA).
            series2: Second series (e.g., slow MA).

        Returns:
            Boolean series that is True when series1 crosses below series2.
        """
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))
