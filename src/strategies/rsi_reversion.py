"""
RSI Mean Reversion Strategy.

This strategy is based on the Relative Strength Index (RSI), which measures
the speed and magnitude of price movements. The core idea is mean reversion:
- When RSI is very low (oversold), prices may bounce back up
- When RSI is very high (overbought), prices may pull back down

This is a counter-trend strategy that works well in ranging markets but
may underperform in strong trending markets.
"""

from typing import Any

import pandas as pd

from .base import Signal, Strategy


class RSIMeanReversionStrategy(Strategy):
    """
    RSI Mean Reversion trading strategy.

    This strategy generates buy signals when RSI falls below an oversold
    threshold (indicating potential reversal up) and sell signals when
    RSI rises above an overbought threshold.

    Attributes:
        period: RSI calculation period.
        oversold: RSI threshold below which to buy (default: 30).
        overbought: RSI threshold above which to sell (default: 70).

    Example:
        >>> strategy = RSIMeanReversionStrategy(period=14, oversold=30, overbought=70)
        >>> signals = strategy.generate_signals(ohlcv_df)

    Note:
        This strategy exits positions when RSI returns to neutral territory
        (crosses back through the thresholds), not at the opposite extreme.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70
    ):
        """
        Initialize the RSI Mean Reversion strategy.

        Args:
            period: RSI calculation period (default: 14).
            oversold: RSI level below which to generate buy signals (default: 30).
            overbought: RSI level above which to generate sell signals (default: 70).

        Raises:
            ValueError: If oversold >= overbought or thresholds out of range.
        """
        super().__init__(name="RSI Mean Reversion")

        if oversold >= overbought:
            raise ValueError(
                f"oversold ({oversold}) must be less than overbought ({overbought})"
            )
        if not (0 < oversold < 100) or not (0 < overbought < 100):
            raise ValueError("RSI thresholds must be between 0 and 100")

        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on RSI levels.

        Buy when RSI crosses below oversold threshold.
        Sell when RSI crosses above overbought threshold.

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            Series of Signal values.
        """
        close = df["close"]

        # Calculate RSI
        rsi = self.rsi(close, self.period)

        # Initialize signals as HOLD
        signals = pd.Series(Signal.HOLD, index=df.index)

        # Generate signals based on RSI levels
        # Buy when RSI crosses into oversold territory
        buy_condition = (rsi < self.oversold) & (rsi.shift(1) >= self.oversold)
        signals[buy_condition] = Signal.BUY

        # Sell when RSI crosses into overbought territory
        sell_condition = (rsi > self.overbought) & (rsi.shift(1) <= self.overbought)
        signals[sell_condition] = Signal.SELL

        return signals

    def get_params(self) -> dict[str, Any]:
        """Return strategy parameters."""
        return {
            "period": self.period,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }
