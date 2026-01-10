"""
Backtesting Framework - A modular system for testing trading strategies.

This package provides tools for:
- Fetching historical OHLCV data from Binance
- Implementing custom trading strategies
- Running backtests with realistic fee simulation
- Calculating performance metrics
- Visualizing results
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .backtest import BacktestEngine
from .data import DataFetcher
from .metrics import PerformanceMetrics
from .visualization import ChartGenerator

__all__ = [
    "BacktestEngine",
    "DataFetcher",
    "PerformanceMetrics",
    "ChartGenerator",
]
