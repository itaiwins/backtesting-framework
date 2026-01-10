"""
MACD (Moving Average Convergence Divergence) Strategy.

MACD is a trend-following momentum indicator that shows the relationship
between two EMAs of the price. This strategy generates signals based on:
1. MACD line crossing above/below the signal line
2. Optionally, histogram direction changes

The MACD is calculated as:
- MACD Line = 12-period EMA - 26-period EMA
- Signal Line = 9-period EMA of MACD Line
- Histogram = MACD Line - Signal Line
"""

from typing import Any

import pandas as pd

from .base import Signal, Strategy


class MACDStrategy(Strategy):
    """
    MACD trading strategy.

    This strategy generates buy signals when the MACD line crosses above
    the signal line, and sell signals when it crosses below.

    Attributes:
        fast_period: Period for the fast EMA (default: 12).
        slow_period: Period for the slow EMA (default: 26).
        signal_period: Period for the signal line EMA (default: 9).

    Example:
        >>> strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
        >>> signals = strategy.generate_signals(ohlcv_df)

    Note:
        Some traders add additional filters like:
        - Only trade when MACD is above/below zero
        - Wait for histogram confirmation
        - Use divergence between price and MACD

        This implementation uses the basic crossover approach for simplicity.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        """
        Initialize the MACD strategy.

        Args:
            fast_period: Period for the fast EMA (default: 12).
            slow_period: Period for the slow EMA (default: 26).
            signal_period: Period for the signal line (default: 9).

        Raises:
            ValueError: If fast_period >= slow_period.
        """
        super().__init__(name="MACD")

        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than "
                f"slow_period ({slow_period})"
            )

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MACD crossovers.

        Buy when MACD line crosses above signal line (bullish).
        Sell when MACD line crosses below signal line (bearish).

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            Series of Signal values.
        """
        close = df["close"]

        # Calculate MACD components
        macd_line, signal_line, histogram = self.macd(
            close,
            self.fast_period,
            self.slow_period,
            self.signal_period
        )

        # Initialize signals as HOLD
        signals = pd.Series(Signal.HOLD, index=df.index)

        # Generate signals on MACD/Signal line crossovers
        signals[self.crossover(macd_line, signal_line)] = Signal.BUY
        signals[self.crossunder(macd_line, signal_line)] = Signal.SELL

        return signals

    def get_params(self) -> dict[str, Any]:
        """Return strategy parameters."""
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
        }
