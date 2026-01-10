"""
Visualization module for generating backtest charts.

This module creates professional-looking charts to visualize
backtest results, including equity curves, drawdowns, and trade markers.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


class ChartGenerator:
    """
    Generates visualizations for backtest results.

    Provides methods to create equity curves, drawdown charts,
    and combined performance dashboards.
    """

    # Color scheme
    COLORS = {
        "equity": "#2E86AB",       # Blue
        "benchmark": "#A23B72",    # Magenta
        "drawdown": "#E94F37",     # Red
        "buy": "#2ECC71",          # Green
        "sell": "#E74C3C",         # Red
        "grid": "#E5E5E5",         # Light gray
        "background": "#FFFFFF",   # White
    }

    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        """
        Initialize the chart generator.

        Args:
            style: Matplotlib style to use.
        """
        try:
            plt.style.use(style)
        except OSError:
            # Fallback for older matplotlib versions
            plt.style.use("seaborn-whitegrid")

    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None,
        title: str = "Equity Curve",
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot the equity curve with optional benchmark and trade markers.

        Args:
            equity_curve: Series of portfolio values indexed by date.
            benchmark: Optional benchmark equity curve (e.g., buy & hold).
            trades: Optional DataFrame with trade entry/exit points.
            title: Chart title.
            save_path: Path to save the figure (optional).
            show: Whether to display the plot.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot equity curve
        ax.plot(
            equity_curve.index,
            equity_curve.values,
            label="Strategy",
            color=self.COLORS["equity"],
            linewidth=2
        )

        # Plot benchmark if provided
        if benchmark is not None:
            ax.plot(
                benchmark.index,
                benchmark.values,
                label="Buy & Hold",
                color=self.COLORS["benchmark"],
                linewidth=1.5,
                linestyle="--",
                alpha=0.8
            )

        # Mark trades if provided
        if trades is not None and len(trades) > 0:
            # Entry points (buys)
            buys = trades[trades["type"] == "entry"]
            if len(buys) > 0:
                buy_dates = pd.to_datetime(buys["date"])
                buy_values = equity_curve.reindex(buy_dates, method="nearest")
                ax.scatter(
                    buy_values.index,
                    buy_values.values,
                    marker="^",
                    color=self.COLORS["buy"],
                    s=100,
                    label="Buy",
                    zorder=5
                )

            # Exit points (sells)
            sells = trades[trades["type"] == "exit"]
            if len(sells) > 0:
                sell_dates = pd.to_datetime(sells["date"])
                sell_values = equity_curve.reindex(sell_dates, method="nearest")
                ax.scatter(
                    sell_values.index,
                    sell_values.values,
                    marker="v",
                    color=self.COLORS["sell"],
                    s=100,
                    label="Sell",
                    zorder=5
                )

        # Formatting
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        # Format y-axis with dollar signs
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Equity curve saved to: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_drawdown(
        self,
        equity_curve: pd.Series,
        title: str = "Drawdown",
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot the drawdown chart.

        Args:
            equity_curve: Series of portfolio values indexed by date.
            title: Chart title.
            save_path: Path to save the figure (optional).
            show: Whether to display the plot.

        Returns:
            Matplotlib Figure object.
        """
        # Calculate drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100

        fig, ax = plt.subplots(figsize=(14, 4))

        # Fill drawdown area
        ax.fill_between(
            drawdown.index,
            drawdown.values,
            0,
            color=self.COLORS["drawdown"],
            alpha=0.3
        )

        ax.plot(
            drawdown.index,
            drawdown.values,
            color=self.COLORS["drawdown"],
            linewidth=1
        )

        # Formatting
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        # Highlight max drawdown
        min_idx = drawdown.idxmin()
        min_val = drawdown.min()
        ax.annotate(
            f"Max: {min_val:.1f}%",
            xy=(min_idx, min_val),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=10,
            color=self.COLORS["drawdown"],
            fontweight="bold"
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Drawdown chart saved to: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_dashboard(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None,
        strategy_name: str = "Strategy",
        ticker: str = "",
        metrics: Optional[dict] = None,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple charts.

        Args:
            equity_curve: Series of portfolio values indexed by date.
            benchmark: Optional benchmark equity curve.
            trades: Optional DataFrame with trade points.
            strategy_name: Name of the strategy.
            ticker: Ticker symbol.
            metrics: Dictionary of performance metrics.
            save_path: Path to save the figure (optional).
            show: Whether to display the plot.

        Returns:
            Matplotlib Figure object.
        """
        fig = plt.figure(figsize=(16, 12))

        # Create grid for subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

        # Main equity curve (spans full width)
        ax1 = fig.add_subplot(gs[0, :])

        # Plot equity
        ax1.plot(
            equity_curve.index,
            equity_curve.values,
            label="Strategy",
            color=self.COLORS["equity"],
            linewidth=2
        )

        if benchmark is not None:
            ax1.plot(
                benchmark.index,
                benchmark.values,
                label="Buy & Hold",
                color=self.COLORS["benchmark"],
                linewidth=1.5,
                linestyle="--",
                alpha=0.8
            )

        ax1.set_title(
            f"{strategy_name} - {ticker}",
            fontsize=16,
            fontweight="bold"
        )
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Drawdown chart
        ax2 = fig.add_subplot(gs[1, :])

        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100

        ax2.fill_between(
            drawdown.index,
            drawdown.values,
            0,
            color=self.COLORS["drawdown"],
            alpha=0.3
        )
        ax2.plot(
            drawdown.index,
            drawdown.values,
            color=self.COLORS["drawdown"],
            linewidth=1
        )
        ax2.set_title("Drawdown", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Monthly returns heatmap
        ax3 = fig.add_subplot(gs[2, 0])

        returns = equity_curve.pct_change().dropna()

        # Create monthly returns table
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly_df = monthly.to_frame(name="return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month

        try:
            pivot = monthly_df.pivot(
                index="year",
                columns="month",
                values="return"
            )

            # Plot heatmap
            im = ax3.imshow(
                pivot.values * 100,
                cmap="RdYlGn",
                aspect="auto",
                vmin=-20,
                vmax=20
            )

            # Labels
            ax3.set_xticks(range(12))
            ax3.set_xticklabels(
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                fontsize=8
            )
            ax3.set_yticks(range(len(pivot.index)))
            ax3.set_yticklabels(pivot.index)
            ax3.set_title("Monthly Returns (%)", fontsize=14, fontweight="bold")

            # Add colorbar
            plt.colorbar(im, ax=ax3, shrink=0.8)
        except Exception:
            ax3.text(
                0.5, 0.5,
                "Insufficient data\nfor monthly returns",
                ha="center", va="center",
                transform=ax3.transAxes
            )
            ax3.set_title("Monthly Returns", fontsize=14, fontweight="bold")

        # Metrics summary
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis("off")

        if metrics:
            metrics_text = (
                f"Performance Summary\n"
                f"{'─' * 30}\n\n"
                f"Total Return:    {metrics.get('total_return', 0):.2%}\n"
                f"Buy & Hold:      {metrics.get('buy_hold_return', 0):.2%}\n"
                f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}\n"
                f"Max Drawdown:    {metrics.get('max_drawdown', 0):.2%}\n"
                f"Win Rate:        {metrics.get('win_rate', 0):.2%}\n"
                f"Total Trades:    {metrics.get('total_trades', 0)}\n"
                f"Profit Factor:   {metrics.get('profit_factor', 0):.2f}"
            )
            ax4.text(
                0.1, 0.9,
                metrics_text,
                transform=ax4.transAxes,
                fontsize=12,
                fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Dashboard saved to: {save_path}")

        if show:
            plt.show()

        return fig


def generate_backtest_charts(
    equity_curve: pd.Series,
    benchmark: pd.Series,
    trades: pd.DataFrame,
    strategy_name: str,
    ticker: str,
    metrics: dict,
    output_dir: str = "output"
) -> dict[str, str]:
    """
    Generate all standard backtest charts.

    Args:
        equity_curve: Strategy equity curve.
        benchmark: Benchmark equity curve.
        trades: DataFrame with trade information.
        strategy_name: Name of the strategy.
        ticker: Ticker symbol.
        metrics: Dictionary of performance metrics.
        output_dir: Directory to save charts.

    Returns:
        Dictionary mapping chart names to file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    chart_gen = ChartGenerator()
    file_base = f"{ticker.lower()}_{strategy_name.lower().replace(' ', '_')}"

    charts = {}

    # Equity curve
    equity_path = str(output_path / f"{file_base}_equity.png")
    chart_gen.plot_equity_curve(
        equity_curve,
        benchmark=benchmark,
        trades=trades,
        title=f"{strategy_name} - {ticker} Equity Curve",
        save_path=equity_path
    )
    charts["equity"] = equity_path

    # Dashboard
    dashboard_path = str(output_path / f"{file_base}_dashboard.png")
    chart_gen.plot_dashboard(
        equity_curve,
        benchmark=benchmark,
        trades=trades,
        strategy_name=strategy_name,
        ticker=ticker,
        metrics=metrics,
        save_path=dashboard_path
    )
    charts["dashboard"] = dashboard_path

    plt.close("all")

    return charts
