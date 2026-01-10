# Backtesting Framework

A clean, modular backtesting framework for testing cryptocurrency trading strategies on historical data. Built with Python, featuring an extensible strategy system, comprehensive performance metrics, and professional visualizations.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Pluggable Strategy System**: Easy-to-extend base class for implementing custom strategies
- **Real-Time Data**: Fetches historical OHLCV data from Binance public API (no API key required)
- **Comprehensive Metrics**: Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor, and more
- **Professional Visualizations**: Equity curves, drawdown charts, and performance dashboards
- **Realistic Simulation**: Includes trading fees and slippage modeling
- **CLI Interface**: Run backtests from the command line
- **Type Hints**: Full type annotations for better IDE support and code clarity

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [How Backtesting Works](#how-backtesting-works)
- [Available Strategies](#available-strategies)
- [Creating Custom Strategies](#creating-custom-strategies)
- [Understanding the Metrics](#understanding-the-metrics)
- [Project Structure](#project-structure)
- [Sample Results](#sample-results)
- [Limitations & Disclaimer](#limitations--disclaimer)

## Installation

```bash
# Clone the repository
git clone https://github.com/itaiwins/backtesting-framework.git
cd backtesting-framework

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- requests >= 2.31.0

## Quick Start

### Python API

```python
from src.backtest import run_backtest
from src.strategies import SMACrossoverStrategy

# Create a strategy
strategy = SMACrossoverStrategy(fast_period=20, slow_period=50)

# Run backtest
result = run_backtest(
    strategy=strategy,
    ticker="BTC",
    start_date="2024-01-01",
    end_date="2024-12-31",
    initial_capital=10000,
    fee_rate=0.001  # 0.1% per trade
)

# Print results
result.print_report()
```

### Command Line

```bash
# Run SMA crossover backtest on BTC
python -m src.cli backtest --strategy sma_cross --ticker BTC --start 2024-01-01 --end 2024-12-31

# Run RSI strategy on ETH with custom fee
python -m src.cli backtest --strategy rsi --ticker ETH --start 2024-06-01 --end 2024-12-31 --fee 0.002

# List available strategies
python -m src.cli list-strategies

# Fetch historical data
python -m src.cli fetch --ticker BTC --start 2024-01-01 --end 2024-03-01
```

## CLI Usage

### Backtest Command

```bash
python -m src.cli backtest [OPTIONS]

Required:
  --strategy, -s    Strategy name (sma_cross, rsi, macd)
  --ticker, -t      Ticker symbol (BTC, ETH, SOL, etc.)
  --start           Start date (YYYY-MM-DD)

Optional:
  --end             End date (defaults to today)
  --capital         Initial capital (default: 10000)
  --fee             Fee rate per trade (default: 0.001)
  --interval        Data interval (default: 1d)
  --no-charts       Skip chart generation
  --output          Output directory for charts

Strategy Parameters:
  --fast-period     Fast MA period (SMA/MACD)
  --slow-period     Slow MA period (SMA/MACD)
  --rsi-period      RSI calculation period
  --oversold        RSI oversold threshold
  --overbought      RSI overbought threshold
```

### Example Output

```
==================================================
          BACKTEST RESULTS
==================================================
Strategy:           SMA Crossover(fast_period=20, slow_period=50)
Ticker:             BTCUSDT
Period:             2024-01-01 to 2024-12-31
--------------------------------------------------

RETURNS
  Total Return:          45.23%
  Buy & Hold Return:     52.10%
  Final Equity:      $14,523.00

RISK METRICS
  Sharpe Ratio:            1.24
  Sortino Ratio:           1.87
  Max Drawdown:          -18.70%

TRADE STATISTICS
  Total Trades:              24
  Win Rate:               58.33%
  Profit Factor:           1.85
  Avg Trade Return:        1.89%
  Avg Win:                 4.52%
  Avg Loss:               -2.31%

STREAKS
  Max Consecutive Wins:       5
  Max Consecutive Losses:     3
==================================================
```

## How Backtesting Works

Backtesting simulates how a trading strategy would have performed on historical data:

```
┌─────────────────────────────────────────────────────────────────┐
│                    BACKTESTING WORKFLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA LOADING                                                │
│     └── Fetch historical OHLCV data from Binance API            │
│                          │                                      │
│                          ▼                                      │
│  2. SIGNAL GENERATION                                           │
│     └── Strategy analyzes data and produces BUY/SELL signals    │
│                          │                                      │
│                          ▼                                      │
│  3. POSITION MANAGEMENT                                         │
│     └── Convert signals to position states (in/out of market)   │
│                          │                                      │
│                          ▼                                      │
│  4. TRADE SIMULATION                                            │
│     └── Execute trades with fees and slippage                   │
│                          │                                      │
│                          ▼                                      │
│  5. PERFORMANCE CALCULATION                                     │
│     └── Calculate equity curve, metrics, and statistics         │
│                          │                                      │
│                          ▼                                      │
│  6. VISUALIZATION                                               │
│     └── Generate charts and reports                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Concepts

- **OHLCV Data**: Open, High, Low, Close, Volume - standard candlestick data
- **Signal**: A trading instruction (BUY, SELL, or HOLD)
- **Position**: Whether you're currently in the market (1) or out (0)
- **Equity Curve**: The value of your portfolio over time
- **Drawdown**: The decline from a peak to a trough in portfolio value

## Available Strategies

### 1. SMA Crossover (`sma_cross`)

A trend-following strategy using two Simple Moving Averages.

```python
from src.strategies import SMACrossoverStrategy

strategy = SMACrossoverStrategy(
    fast_period=20,  # Short-term MA
    slow_period=50   # Long-term MA
)
```

**Logic:**
- **BUY** when fast SMA crosses above slow SMA (Golden Cross)
- **SELL** when fast SMA crosses below slow SMA (Death Cross)

### 2. RSI Mean Reversion (`rsi`)

A mean reversion strategy using the Relative Strength Index.

```python
from src.strategies import RSIMeanReversionStrategy

strategy = RSIMeanReversionStrategy(
    period=14,       # RSI calculation period
    oversold=30,     # Buy threshold
    overbought=70    # Sell threshold
)
```

**Logic:**
- **BUY** when RSI drops below oversold threshold (market may be undervalued)
- **SELL** when RSI rises above overbought threshold (market may be overvalued)

### 3. MACD (`macd`)

A momentum strategy using Moving Average Convergence Divergence.

```python
from src.strategies import MACDStrategy

strategy = MACDStrategy(
    fast_period=12,    # Fast EMA period
    slow_period=26,    # Slow EMA period
    signal_period=9    # Signal line period
)
```

**Logic:**
- **BUY** when MACD line crosses above signal line (bullish momentum)
- **SELL** when MACD line crosses below signal line (bearish momentum)

## Creating Custom Strategies

Implementing your own strategy is straightforward:

```python
from src.strategies.base import Strategy, Signal
import pandas as pd

class MyCustomStrategy(Strategy):
    """
    My awesome trading strategy.

    Buy when price is above the 50-day high.
    Sell when price drops below the 20-day low.
    """

    def __init__(self, high_period: int = 50, low_period: int = 20):
        super().__init__(name="Breakout Strategy")
        self.high_period = high_period
        self.low_period = low_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns

        Returns:
            Series with Signal.BUY (1), Signal.SELL (-1), or Signal.HOLD (0)
        """
        # Calculate indicators
        high_channel = df['high'].rolling(self.high_period).max()
        low_channel = df['low'].rolling(self.low_period).min()

        # Initialize all signals as HOLD
        signals = pd.Series(Signal.HOLD, index=df.index)

        # Generate signals
        signals[df['close'] > high_channel.shift(1)] = Signal.BUY
        signals[df['close'] < low_channel.shift(1)] = Signal.SELL

        return signals

    def get_params(self) -> dict:
        """Return parameters for reporting."""
        return {
            "high_period": self.high_period,
            "low_period": self.low_period
        }

# Use your strategy
from src.backtest import run_backtest

result = run_backtest(
    strategy=MyCustomStrategy(high_period=50, low_period=20),
    ticker="BTC",
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

### Available Helper Methods

The base `Strategy` class provides these technical indicator helpers:

| Method | Description |
|--------|-------------|
| `sma(series, period)` | Simple Moving Average |
| `ema(series, period)` | Exponential Moving Average |
| `rsi(series, period)` | Relative Strength Index |
| `macd(series, fast, slow, signal)` | MACD indicator |
| `bollinger_bands(series, period, std)` | Bollinger Bands |
| `atr(high, low, close, period)` | Average True Range |
| `crossover(series1, series2)` | Detect bullish crossover |
| `crossunder(series1, series2)` | Detect bearish crossunder |

## Understanding the Metrics

### Return Metrics

| Metric | Description | Good Values |
|--------|-------------|-------------|
| **Total Return** | Overall profit/loss percentage | Depends on market |
| **Buy & Hold Return** | Benchmark: holding the asset | Compare to Total Return |

### Risk Metrics

| Metric | Description | Good Values |
|--------|-------------|-------------|
| **Sharpe Ratio** | Risk-adjusted return (return per unit of volatility) | > 1.0 is good, > 2.0 is excellent |
| **Sortino Ratio** | Like Sharpe, but only penalizes downside volatility | > 1.5 is good |
| **Max Drawdown** | Largest peak-to-trough decline | < -20% is concerning |

### Trade Statistics

| Metric | Description | Good Values |
|--------|-------------|-------------|
| **Win Rate** | Percentage of profitable trades | > 50% with proper risk/reward |
| **Profit Factor** | Gross profits / Gross losses | > 1.5 is good, > 2.0 is excellent |
| **Avg Trade Return** | Mean return per trade | Positive |

## Project Structure

```
backtesting-framework/
├── src/
│   ├── __init__.py          # Package exports
│   ├── backtest.py          # Main backtesting engine
│   ├── data.py              # Data fetching from Binance
│   ├── metrics.py           # Performance calculations
│   ├── visualization.py     # Chart generation
│   ├── cli.py               # Command-line interface
│   └── strategies/
│       ├── __init__.py      # Strategy registry
│       ├── base.py          # Base Strategy class
│       ├── sma_cross.py     # SMA Crossover strategy
│       ├── rsi_reversion.py # RSI Mean Reversion strategy
│       └── macd.py          # MACD strategy
├── examples/
│   └── run_backtest.py      # Example usage scripts
├── output/                  # Generated charts
├── requirements.txt         # Dependencies
├── .gitignore
└── README.md
```

## Sample Results

### Equity Curve

The equity curve shows your portfolio value over time compared to buy & hold:

```
Portfolio Value ($)
    │
15k ┤                                    ╭──────────╮
    │                              ╭─────╯          │  Strategy
14k ┤                         ╭────╯                │
    │                    ╭────╯                     │
13k ┤               ╭────╯ ╱╲╱╲                     │
    │          ╭────╯    ╱    ╲                     │
12k ┤     ╭────╯       ╱       ╲ Buy & Hold         │
    │ ────╯           ╱                             │
11k ┤               ╱                               │
    │              ╱                                │
10k ┼─────────────┴────────────────────────────────┴──►
    Jan     Mar     May     Jul     Sep     Nov   Date
```

### Drawdown Chart

Shows the decline from peak portfolio value:

```
Drawdown (%)
  0% ┬──────╮     ╭──────────╮      ╭────────────────
     │      │     │          │      │
 -5% ┤      │     │          │      │
     │      ╰─────╯          │      │
-10% ┤                       │      │
     │                       ╰──────╯
-15% ┤
     │          Max: -18.7%
-20% ┴─────────────────────────────────────────────►
```

## Limitations & Disclaimer

### Current Limitations

1. **Long-Only**: The framework currently supports long positions only (no short selling)
2. **Single Asset**: Backtests run on one asset at a time (no portfolio optimization)
3. **No Order Book**: Uses close prices; doesn't simulate limit orders or order book dynamics
4. **Point-in-Time**: Signals are generated on close prices (potential look-ahead bias if misused)
5. **Daily Data Default**: While multiple intervals are supported, daily data is most reliable

### Disclaimer

⚠️ **This software is for educational and research purposes only.**

- Past performance does not guarantee future results
- Backtests often overestimate real-world performance
- Real trading involves risks including total loss of capital
- This is NOT financial advice
- Always do your own research and consider consulting a financial advisor

### Known Biases in Backtesting

| Bias | Description |
|------|-------------|
| **Survivorship Bias** | Only testing on assets that still exist |
| **Look-Ahead Bias** | Using information not available at the time |
| **Overfitting** | Optimizing parameters to fit historical data |
| **Transaction Costs** | Underestimating fees, slippage, and market impact |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Built with Python | Data from [Binance](https://www.binance.com/)
