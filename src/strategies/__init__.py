"""
Trading strategies module.

This module contains the base Strategy class and pre-built strategy implementations.
To create a custom strategy, inherit from the Strategy base class and implement
the generate_signals() method.
"""

from .base import Strategy, Signal
from .sma_cross import SMACrossoverStrategy
from .rsi_reversion import RSIMeanReversionStrategy
from .macd import MACDStrategy

__all__ = [
    "Strategy",
    "Signal",
    "SMACrossoverStrategy",
    "RSIMeanReversionStrategy",
    "MACDStrategy",
]

# Registry of available strategies for CLI
STRATEGY_REGISTRY = {
    "sma_cross": SMACrossoverStrategy,
    "sma": SMACrossoverStrategy,
    "rsi": RSIMeanReversionStrategy,
    "rsi_reversion": RSIMeanReversionStrategy,
    "macd": MACDStrategy,
}


def get_strategy(name: str) -> type:
    """
    Get a strategy class by name.

    Args:
        name: Strategy name (e.g., "sma_cross", "rsi", "macd")

    Returns:
        Strategy class.

    Raises:
        ValueError: If strategy name is not found.
    """
    name = name.lower()
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy '{name}'. Available strategies: {available}"
        )
    return STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """Get list of available strategy names."""
    return list(set(STRATEGY_REGISTRY.values()))
