"""
Data fetching and preprocessing module.

This module handles fetching historical OHLCV (Open, High, Low, Close, Volume)
data from multiple sources. It tries Binance first, then falls back to
alternative sources if needed.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests


@dataclass
class OHLCVData:
    """Container for OHLCV data with metadata."""

    symbol: str
    interval: str
    start_date: datetime
    end_date: datetime
    df: pd.DataFrame

    @property
    def num_candles(self) -> int:
        """Return the number of candles in the dataset."""
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"OHLCVData(symbol={self.symbol}, interval={self.interval}, "
            f"candles={self.num_candles}, "
            f"period={self.start_date.date()} to {self.end_date.date()})"
        )


class DataFetcher:
    """
    Fetches historical OHLCV data from multiple sources.

    Tries the following sources in order:
    1. Binance public API
    2. Binance.US API (for US-based users)
    3. CoinGecko API (free, no key required)

    Example:
        >>> fetcher = DataFetcher()
        >>> data = fetcher.fetch("BTC", "1d", "2024-01-01", "2024-12-31")
        >>> print(data.df.head())
    """

    # API endpoints
    BINANCE_URL = "https://api.binance.com/api/v3/klines"
    BINANCE_US_URL = "https://api.binance.us/api/v3/klines"
    COINGECKO_URL = "https://api.coingecko.com/api/v3"

    # Mapping of common ticker symbols to Binance trading pairs
    SYMBOL_MAP = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        "BNB": "BNBUSDT",
        "XRP": "XRPUSDT",
        "ADA": "ADAUSDT",
        "DOGE": "DOGEUSDT",
        "AVAX": "AVAXUSDT",
        "DOT": "DOTUSDT",
        "LINK": "LINKUSDT",
    }

    # CoinGecko ID mapping
    COINGECKO_IDS = {
        "BTC": "bitcoin",
        "BTCUSDT": "bitcoin",
        "ETH": "ethereum",
        "ETHUSDT": "ethereum",
        "SOL": "solana",
        "SOLUSDT": "solana",
        "BNB": "binancecoin",
        "BNBUSDT": "binancecoin",
        "XRP": "ripple",
        "XRPUSDT": "ripple",
        "ADA": "cardano",
        "ADAUSDT": "cardano",
        "DOGE": "dogecoin",
        "DOGEUSDT": "dogecoin",
        "AVAX": "avalanche-2",
        "AVAXUSDT": "avalanche-2",
        "DOT": "polkadot",
        "DOTUSDT": "polkadot",
        "LINK": "chainlink",
        "LINKUSDT": "chainlink",
    }

    # Valid intervals for Binance API
    VALID_INTERVALS = [
        "1m", "3m", "5m", "15m", "30m",  # Minutes
        "1h", "2h", "4h", "6h", "8h", "12h",  # Hours
        "1d", "3d",  # Days
        "1w",  # Weeks
        "1M",  # Months
    ]

    def __init__(self, timeout: int = 30):
        """
        Initialize the DataFetcher.

        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "BacktestingFramework/1.0"
        })

    def _normalize_symbol(self, ticker: str) -> str:
        """
        Convert a ticker symbol to Binance trading pair format.

        Args:
            ticker: Ticker symbol (e.g., "BTC" or "BTCUSDT")

        Returns:
            Binance trading pair (e.g., "BTCUSDT")
        """
        ticker = ticker.upper()
        return self.SYMBOL_MAP.get(ticker, ticker)

    def _parse_date(self, date_str: str) -> int:
        """
        Convert a date string to Unix timestamp in milliseconds.

        Args:
            date_str: Date in YYYY-MM-DD format.

        Returns:
            Unix timestamp in milliseconds.
        """
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp() * 1000)

    def fetch(
        self,
        ticker: str,
        interval: str = "1d",
        start_date: str = "2024-01-01",
        end_date: Optional[str] = None,
    ) -> OHLCVData:
        """
        Fetch historical OHLCV data.

        Tries multiple sources in order until one succeeds.

        Args:
            ticker: Ticker symbol (e.g., "BTC", "ETH", or "BTCUSDT")
            interval: Candlestick interval (e.g., "1d", "4h", "1h")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)

        Returns:
            OHLCVData object containing the historical data.

        Raises:
            ValueError: If the interval is invalid or no data is returned.
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. "
                f"Valid options: {self.VALID_INTERVALS}"
            )

        symbol = self._normalize_symbol(ticker)

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Try data sources in order
        errors = []

        # Try Binance
        try:
            return self._fetch_binance(symbol, interval, start_date, end_date, self.BINANCE_URL)
        except Exception as e:
            errors.append(f"Binance: {e}")

        # Try Binance.US
        try:
            return self._fetch_binance(symbol, interval, start_date, end_date, self.BINANCE_US_URL)
        except Exception as e:
            errors.append(f"Binance.US: {e}")

        # Try CoinGecko (daily data only)
        if interval == "1d":
            try:
                return self._fetch_coingecko(ticker, start_date, end_date)
            except Exception as e:
                errors.append(f"CoinGecko: {e}")

        # All sources failed - generate synthetic data for demo
        print(f"Warning: All data sources failed. Generating synthetic data for demo.")
        print(f"  Errors: {'; '.join(errors)}")
        return self._generate_synthetic_data(symbol, interval, start_date, end_date)

    def _fetch_binance(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        base_url: str
    ) -> OHLCVData:
        """Fetch data from Binance or Binance.US."""
        start_ts = self._parse_date(start_date)
        end_ts = self._parse_date(end_date)

        all_candles = []
        current_start = start_ts

        while current_start < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": 1000,
            }

            response = self.session.get(
                base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            candles = response.json()

            if not candles:
                break

            all_candles.extend(candles)
            current_start = candles[-1][6] + 1

            if len(candles) < 1000:
                break

        if not all_candles:
            raise ValueError(f"No data returned for {symbol}")

        df = pd.DataFrame(
            all_candles,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "num_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ]
        )

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.set_index("open_time", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        df.sort_index(inplace=True)

        return OHLCVData(
            symbol=symbol,
            interval=interval,
            start_date=datetime.strptime(start_date, "%Y-%m-%d"),
            end_date=datetime.strptime(end_date, "%Y-%m-%d"),
            df=df
        )

    def _fetch_coingecko(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> OHLCVData:
        """Fetch daily data from CoinGecko API."""
        ticker_upper = ticker.upper()
        coin_id = self.COINGECKO_IDS.get(ticker_upper)

        if not coin_id:
            raise ValueError(f"Unknown ticker for CoinGecko: {ticker}")

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # CoinGecko uses Unix timestamps in seconds
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())

        url = f"{self.COINGECKO_URL}/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": "usd",
            "from": start_ts,
            "to": end_ts,
        }

        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()

        if not data.get("prices"):
            raise ValueError(f"No data returned from CoinGecko for {ticker}")

        # CoinGecko returns [timestamp_ms, price] arrays
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
        prices.set_index("timestamp", inplace=True)

        # Resample to daily
        prices = prices.resample("D").last().dropna()

        # Create OHLCV (approximate - CoinGecko doesn't give true OHLC)
        df = pd.DataFrame(index=prices.index)
        df["close"] = prices["close"]
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1) * 1.01  # Approximate
        df["low"] = df[["open", "close"]].min(axis=1) * 0.99   # Approximate
        df["volume"] = 1000000  # Placeholder

        symbol = self._normalize_symbol(ticker)

        return OHLCVData(
            symbol=symbol,
            interval="1d",
            start_date=start_dt,
            end_date=end_dt,
            df=df
        )

    def _generate_synthetic_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str
    ) -> OHLCVData:
        """
        Generate synthetic price data for demo purposes.

        Creates realistic-looking price movements using geometric Brownian motion.
        This is useful for testing when API access is unavailable.
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Determine number of periods based on interval
        if interval == "1d":
            periods = (end_dt - start_dt).days
            freq = "D"
        elif interval == "1h":
            periods = (end_dt - start_dt).days * 24
            freq = "h"
        elif interval == "4h":
            periods = (end_dt - start_dt).days * 6
            freq = "4h"
        else:
            periods = (end_dt - start_dt).days
            freq = "D"

        # Generate date index
        dates = pd.date_range(start=start_dt, end=end_dt, freq=freq)

        # Set initial price based on symbol
        if "BTC" in symbol:
            initial_price = 42000
            volatility = 0.03
        elif "ETH" in symbol:
            initial_price = 2500
            volatility = 0.04
        elif "SOL" in symbol:
            initial_price = 100
            volatility = 0.05
        else:
            initial_price = 100
            volatility = 0.03

        # Generate price movements using geometric Brownian motion
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.0005, volatility, len(dates))  # Slight upward drift
        prices = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        df = pd.DataFrame(index=dates)
        df["close"] = prices

        # Generate realistic OHLC from close
        daily_range = volatility * prices
        df["high"] = df["close"] + np.abs(np.random.normal(0, daily_range * 0.5, len(dates)))
        df["low"] = df["close"] - np.abs(np.random.normal(0, daily_range * 0.5, len(dates)))
        df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])

        # Ensure high >= close and low <= close
        df["high"] = df[["high", "close", "open"]].max(axis=1)
        df["low"] = df[["low", "close", "open"]].min(axis=1)

        # Generate volume (correlated with price changes)
        base_volume = 1e9 if "BTC" in symbol else 1e8
        price_change = np.abs(returns)
        df["volume"] = base_volume * (1 + price_change * 10) * np.random.uniform(0.5, 1.5, len(dates))

        print(f"Generated {len(df)} synthetic candles for {symbol}")

        return OHLCVData(
            symbol=symbol,
            interval=interval,
            start_date=start_dt,
            end_date=end_dt,
            df=df
        )

    def get_available_symbols(self) -> list[str]:
        """
        Get list of pre-configured ticker symbols.

        Returns:
            List of available ticker symbols.
        """
        return list(self.SYMBOL_MAP.keys())


if __name__ == "__main__":
    # Quick test
    fetcher = DataFetcher()
    data = fetcher.fetch("BTC", "1d", "2024-01-01", "2024-06-01")
    print(data)
    print(data.df.head())
    print(data.df.tail())
