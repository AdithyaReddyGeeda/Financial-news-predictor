"""
Stock Price Data Fetcher using yfinance.

Downloads OHLCV (Open, High, Low, Close, Volume) data for given tickers
and caches results locally to avoid redundant API calls.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
import yaml

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetch and cache historical stock price data via yfinance."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.cache_dir = self.config["data"].get("cache_dir", "data/raw")
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').
            start_date: Start date string 'YYYY-MM-DD'. Defaults to config.
            end_date: End date string 'YYYY-MM-DD'. Defaults to today.
            use_cache: Whether to use/save cached data.

        Returns:
            DataFrame with columns: [Date, Open, High, Low, Close, Volume, Ticker]
        """
        start = start_date or self.config["data"].get("start_date") or "2020-01-01"
        end = end_date or self.config["data"].get("end_date") or datetime.now().strftime("%Y-%m-%d")

        cache_path = os.path.join(self.cache_dir, f"{ticker}_ohlcv.csv")

        if use_cache and os.path.exists(cache_path):
            cached_df = pd.read_csv(cache_path, parse_dates=["Date"])
            cache_end = cached_df["Date"].max()

            if cache_end >= pd.to_datetime(end) - timedelta(days=1):
                logger.info(f"Using cached data for {ticker}")
                # Filter to requested date range
                mask = cached_df["Date"] >= pd.to_datetime(start)
                return cached_df[mask].reset_index(drop=True)

            logger.info(f"Cache outdated for {ticker}, fetching new data")

        logger.info(f"Downloading {ticker} data: {start} to {end}")

        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            raise

        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'")

        df = df.reset_index()

        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]

        df["Ticker"] = ticker

        expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
        for col in expected_cols:
            if col not in df.columns and col != "Ticker":
                logger.warning(f"Column '{col}' missing from {ticker} data")

        if use_cache:
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached {ticker} data to {cache_path}")

        return df

    def fetch_multiple(
        self,
        tickers: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple tickers.

        Args:
            tickers: List of ticker symbols. Defaults to config defaults.
            start_date: Start date string.
            end_date: End date string.

        Returns:
            Dictionary mapping ticker -> DataFrame.
        """
        tickers = tickers or self.config["data"].get(
            "default_tickers", ["AAPL", "GOOGL", "MSFT"]
        )

        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch(ticker, start_date, end_date)
                logger.info(f"Fetched {len(results[ticker])} rows for {ticker}")
            except Exception as e:
                logger.error(f"Skipping {ticker}: {e}")
                continue

        if not results:
            raise ValueError("Failed to fetch data for any ticker")

        return results

    def get_latest_price(self, ticker: str) -> dict:
        """Get the most recent price data for a ticker."""
        stock = yf.Ticker(ticker)
        info = stock.fast_info

        return {
            "ticker": ticker,
            "last_price": info.get("lastPrice", None),
            "market_cap": info.get("marketCap", None),
            "previous_close": info.get("previousClose", None),
        }
