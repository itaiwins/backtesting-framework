"""
Simple Moving Average (SMA) Crossover Strategy.

This is one of the most classic trend-following strategies. It uses two
moving averages of different periods:
- When the fast MA crosses above the slow MA, it generates a BUY signal
- When the fast MA crosses below the slow MA, it generates a SELL signal

The logic is that when short-term momentum (fast MA) exceeds long-term
momentum (slow MA), the price is trending upward.
"""

from typing import Any

import pandas as pd

from .base import Signal, Strategy


class SMACrossoverStrategy(Strategy):
    """
    SMA Crossover trading strategy.

    This strategy generates buy signals when a faster moving average
    crosses above a slower moving average, and sell signals when
    the faster MA crosses below the slower MA.

    Attributes:
        fast_period: Period for the fast (short-term) SMA.
        slow_period: Period for the slow (long-term) SMA.

    Example:
        >>> strategy = SMACrossoverStrategy(fast_period=10, slow_period=30)
        >>> signals = strategy.generate_signals(ohlcv_df)
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        """
        Initialize the SMA Crossover strategy.

        Args:
            fast_period: Period for the fast SMA (default: 20).
            slow_period: Period for the slow SMA (default: 50).

        Raises:
            ValueError: If fast_period >= slow_period.
        """
        super().__init__(name="SMA Crossover")

        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than "
                f"slow_period ({slow_period})"
            )

        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on SMA crossovers.

        Buy when fast SMA crosses above slow SMA (golden cross).
        Sell when fast SMA crosses below slow SMA (death cross).

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            Series of Signal values.
        """
        close = df["close"]

        # Calculate moving averages
        fast_sma = self.sma(close, self.fast_period)
        slow_sma = self.sma(close, self.slow_period)

        # Initialize signals as HOLD
        signals = pd.Series(Signal.HOLD, index=df.index)

        # Generate signals on crossovers
        signals[self.crossover(fast_sma, slow_sma)] = Signal.BUY
        signals[self.crossunder(fast_sma, slow_sma)] = Signal.SELL

        return signals

    def get_params(self) -> dict[str, Any]:
        """Return strategy parameters."""
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
        }
