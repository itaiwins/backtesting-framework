"""
Performance metrics calculation module.

This module provides functions for calculating various performance metrics
commonly used to evaluate trading strategies. Each metric is explained
in detail to help understand what it measures and how to interpret it.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """
    Container for backtest performance metrics.

    All return values are expressed as decimals (e.g., 0.10 = 10%).
    Monetary values are in the same currency as the initial capital.

    Attributes:
        total_return: Total percentage return of the strategy.
        buy_hold_return: Return from simple buy-and-hold strategy.
        sharpe_ratio: Risk-adjusted return metric.
        sortino_ratio: Downside risk-adjusted return metric.
        max_drawdown: Largest peak-to-trough decline.
        win_rate: Percentage of profitable trades.
        profit_factor: Ratio of gross profits to gross losses.
        total_trades: Number of completed trades.
        avg_trade_return: Average return per trade.
        avg_win: Average return on winning trades.
        avg_loss: Average return on losing trades.
        max_consecutive_wins: Longest winning streak.
        max_consecutive_losses: Longest losing streak.
        final_equity: Final portfolio value.
        initial_capital: Starting capital.
    """

    total_return: float
    buy_hold_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    final_equity: float
    initial_capital: float

    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(\n"
            f"  total_return={self.total_return:.2%},\n"
            f"  sharpe_ratio={self.sharpe_ratio:.2f},\n"
            f"  max_drawdown={self.max_drawdown:.2%},\n"
            f"  win_rate={self.win_rate:.2%},\n"
            f"  total_trades={self.total_trades}\n"
            f")"
        )


def calculate_total_return(equity_curve: pd.Series) -> float:
    """
    Calculate the total percentage return.

    Total Return = (Final Value - Initial Value) / Initial Value

    This is the most basic measure of strategy performance - how much
    did the portfolio grow (or shrink) over the entire period.

    Args:
        equity_curve: Series of portfolio values over time.

    Returns:
        Total return as a decimal (e.g., 0.50 = 50% return).
    """
    if len(equity_curve) < 2:
        return 0.0

    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]

    if initial == 0:
        return 0.0

    return (final - initial) / initial


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe Ratio.

    Sharpe Ratio = (Mean Return - Risk-Free Rate) / Standard Deviation of Returns

    The Sharpe ratio measures risk-adjusted return. It tells you how much
    excess return you receive for the volatility you endure.

    Interpretation:
    - < 0: Strategy loses money on average
    - 0-1: Returns don't compensate well for risk
    - 1-2: Good risk-adjusted returns
    - 2-3: Very good risk-adjusted returns
    - > 3: Excellent (but verify - may be too good to be true)

    Args:
        returns: Series of periodic returns.
        risk_free_rate: Annual risk-free rate (default: 0).
        periods_per_year: Number of trading periods per year (252 for daily).

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0

    # Convert annual risk-free rate to per-period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - rf_per_period
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0:
        return 0.0

    # Annualize the Sharpe ratio
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sortino Ratio.

    Sortino Ratio = (Mean Return - Risk-Free Rate) / Downside Deviation

    Similar to Sharpe ratio, but only penalizes downside volatility.
    This is arguably a better measure since upside volatility is desirable.

    Args:
        returns: Series of periodic returns.
        risk_free_rate: Annual risk-free rate (default: 0).
        periods_per_year: Number of trading periods per year.

    Returns:
        Annualized Sortino ratio.
    """
    if len(returns) < 2:
        return 0.0

    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period

    # Only consider negative returns for downside deviation
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0

    downside_std = np.sqrt((downside_returns ** 2).mean())

    if downside_std == 0:
        return 0.0

    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)

    return sortino


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate the Maximum Drawdown.

    Max Drawdown = (Trough Value - Peak Value) / Peak Value

    This measures the largest peak-to-trough decline in portfolio value.
    It's crucial for understanding the worst-case scenario and helps
    with position sizing decisions.

    Example: A 50% drawdown requires a 100% gain to recover.

    Args:
        equity_curve: Series of portfolio values over time.

    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.20 = -20%).
    """
    if len(equity_curve) < 2:
        return 0.0

    # Calculate running maximum
    running_max = equity_curve.cummax()

    # Calculate drawdown at each point
    drawdown = (equity_curve - running_max) / running_max

    # Return the maximum (most negative) drawdown
    return drawdown.min()


def calculate_win_rate(trade_returns: list[float]) -> float:
    """
    Calculate the Win Rate (percentage of profitable trades).

    Win Rate = Number of Winning Trades / Total Number of Trades

    A win rate alone doesn't determine profitability - you can have
    a 30% win rate and still be profitable if wins are much larger
    than losses.

    Args:
        trade_returns: List of returns for each completed trade.

    Returns:
        Win rate as a decimal (e.g., 0.60 = 60% win rate).
    """
    if not trade_returns:
        return 0.0

    wins = sum(1 for r in trade_returns if r > 0)
    return wins / len(trade_returns)


def calculate_profit_factor(trade_returns: list[float]) -> float:
    """
    Calculate the Profit Factor.

    Profit Factor = Gross Profits / Gross Losses

    This measures how much you make for every dollar you lose.

    Interpretation:
    - < 1.0: Losing strategy
    - 1.0-1.5: Marginal
    - 1.5-2.0: Good
    - > 2.0: Very good

    Args:
        trade_returns: List of returns for each completed trade.

    Returns:
        Profit factor (returns infinity if no losing trades).
    """
    if not trade_returns:
        return 0.0

    gross_profits = sum(r for r in trade_returns if r > 0)
    gross_losses = abs(sum(r for r in trade_returns if r < 0))

    if gross_losses == 0:
        return float('inf') if gross_profits > 0 else 0.0

    return gross_profits / gross_losses


def calculate_consecutive_wins_losses(
    trade_returns: list[float]
) -> tuple[int, int]:
    """
    Calculate maximum consecutive wins and losses.

    These metrics help understand the psychological demands of a strategy.
    Long losing streaks can be difficult to endure even if the strategy
    is ultimately profitable.

    Args:
        trade_returns: List of returns for each completed trade.

    Returns:
        Tuple of (max_consecutive_wins, max_consecutive_losses).
    """
    if not trade_returns:
        return 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for r in trade_returns:
        if r > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif r < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            # Break-even trade
            current_wins = 0
            current_losses = 0

    return max_wins, max_losses


def calculate_metrics(
    equity_curve: pd.Series,
    trade_returns: list[float],
    initial_capital: float,
    buy_hold_return: Optional[float] = None,
    periods_per_year: int = 252
) -> PerformanceMetrics:
    """
    Calculate all performance metrics.

    This is the main function that computes all metrics and returns
    them in a structured format.

    Args:
        equity_curve: Series of portfolio values over time.
        trade_returns: List of returns for each completed trade.
        initial_capital: Starting capital amount.
        buy_hold_return: Pre-calculated buy-and-hold return (optional).
        periods_per_year: Trading periods per year (252 for daily).

    Returns:
        PerformanceMetrics dataclass with all calculated metrics.
    """
    # Calculate periodic returns from equity curve
    returns = equity_curve.pct_change().dropna()

    # Basic returns
    total_return = calculate_total_return(equity_curve)

    # Risk metrics
    sharpe = calculate_sharpe_ratio(returns, periods_per_year=periods_per_year)
    sortino = calculate_sortino_ratio(returns, periods_per_year=periods_per_year)
    max_dd = calculate_max_drawdown(equity_curve)

    # Trade-based metrics
    win_rate = calculate_win_rate(trade_returns)
    profit_factor = calculate_profit_factor(trade_returns)
    max_wins, max_losses = calculate_consecutive_wins_losses(trade_returns)

    # Average trade calculations
    avg_return = np.mean(trade_returns) if trade_returns else 0.0
    winning_trades = [r for r in trade_returns if r > 0]
    losing_trades = [r for r in trade_returns if r < 0]
    avg_win = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = np.mean(losing_trades) if losing_trades else 0.0

    return PerformanceMetrics(
        total_return=total_return,
        buy_hold_return=buy_hold_return if buy_hold_return is not None else 0.0,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=len(trade_returns),
        avg_trade_return=avg_return,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses,
        final_equity=equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital,
        initial_capital=initial_capital,
    )


def format_metrics_report(
    metrics: PerformanceMetrics,
    strategy_name: str,
    ticker: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Format metrics as a human-readable report string.

    Args:
        metrics: Calculated performance metrics.
        strategy_name: Name of the strategy.
        ticker: Trading symbol.
        start_date: Backtest start date.
        end_date: Backtest end date.

    Returns:
        Formatted string report.
    """
    lines = [
        "",
        "=" * 50,
        "          BACKTEST RESULTS",
        "=" * 50,
        f"Strategy:           {strategy_name}",
        f"Ticker:             {ticker}",
        f"Period:             {start_date} to {end_date}",
        "-" * 50,
        "",
        "RETURNS",
        f"  Total Return:       {metrics.total_return:>10.2%}",
        f"  Buy & Hold Return:  {metrics.buy_hold_return:>10.2%}",
        f"  Final Equity:       ${metrics.final_equity:>10,.2f}",
        "",
        "RISK METRICS",
        f"  Sharpe Ratio:       {metrics.sharpe_ratio:>10.2f}",
        f"  Sortino Ratio:      {metrics.sortino_ratio:>10.2f}",
        f"  Max Drawdown:       {metrics.max_drawdown:>10.2%}",
        "",
        "TRADE STATISTICS",
        f"  Total Trades:       {metrics.total_trades:>10}",
        f"  Win Rate:           {metrics.win_rate:>10.2%}",
        f"  Profit Factor:      {metrics.profit_factor:>10.2f}",
        f"  Avg Trade Return:   {metrics.avg_trade_return:>10.2%}",
        f"  Avg Win:            {metrics.avg_win:>10.2%}",
        f"  Avg Loss:           {metrics.avg_loss:>10.2%}",
        "",
        "STREAKS",
        f"  Max Consecutive Wins:   {metrics.max_consecutive_wins:>6}",
        f"  Max Consecutive Losses: {metrics.max_consecutive_losses:>6}",
        "=" * 50,
        "",
    ]

    return "\n".join(lines)
