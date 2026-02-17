"""
Feature Engineering for Stock Price Data.

Computes technical indicators from OHLCV data and merges
with sentiment features to create the final feature matrix
for LSTM input.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ta
import yaml

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute technical indicators and build feature matrices."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.ti_config = self.config.get("technical_indicators", {})
        self.scaler = StandardScaler()
        self._is_fitted = False

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV DataFrame.

        Computes RSI, MACD, Bollinger Bands, OBV, and ATR.

        Args:
            df: DataFrame with columns [Date, Open, High, Low, Close, Volume].

        Returns:
            DataFrame with additional technical indicator columns.
        """
        df = df.copy()
        df = df.sort_values("Date").reset_index(drop=True)

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # RSI (Relative Strength Index)
        rsi_period = self.ti_config.get("rsi_period", 14)
        df["rsi"] = ta.momentum.RSIIndicator(
            close=close, window=rsi_period
        ).rsi()

        # MACD (Moving Average Convergence Divergence)
        macd_fast = self.ti_config.get("macd_fast", 12)
        macd_slow = self.ti_config.get("macd_slow", 26)
        macd_signal = self.ti_config.get("macd_signal", 9)
        macd_indicator = ta.trend.MACD(
            close=close,
            window_slow=macd_slow,
            window_fast=macd_fast,
            window_sign=macd_signal,
        )
        df["macd"] = macd_indicator.macd()
        df["macd_signal_line"] = macd_indicator.macd_signal()
        df["macd_hist"] = macd_indicator.macd_diff()

        # Bollinger Bands
        bb_period = self.ti_config.get("bollinger_period", 20)
        bb_std = self.ti_config.get("bollinger_std", 2)
        bb_indicator = ta.volatility.BollingerBands(
            close=close, window=bb_period, window_dev=bb_std
        )
        df["bb_upper"] = bb_indicator.bollinger_hband()
        df["bb_lower"] = bb_indicator.bollinger_lband()
        df["bb_pct"] = bb_indicator.bollinger_pband()

        # OBV (On-Balance Volume)
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=close, volume=volume
        ).on_balance_volume()

        # ATR (Average True Range)
        atr_period = self.ti_config.get("atr_period", 14)
        df["atr"] = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=atr_period
        ).average_true_range()

        # Price-derived features
        df["returns"] = close.pct_change()
        df["log_returns"] = np.log(close / close.shift(1))
        df["volatility_20d"] = df["returns"].rolling(window=20).std()

        # Targets: next-day direction and return
        df["target_direction"] = (close.shift(-1) > close).astype(int)
        df["target_return"] = close.pct_change().shift(-1)

        logger.info(f"Added technical indicators: {self._get_indicator_columns()}")
        return df

    def merge_sentiment_features(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """
        Merge daily sentiment scores with price/technical indicator data.

        Aligns by date using forward-fill for days without news.

        Args:
            price_df: OHLCV + technical indicators DataFrame with 'Date' column.
            sentiment_df: Daily aggregated sentiment with columns:
                [date, ticker, sentiment_positive, sentiment_negative,
                 sentiment_neutral, sentiment_score]
            ticker: Ticker symbol to filter sentiment data.

        Returns:
            Merged DataFrame with all features aligned by date.
        """
        price_df = price_df.copy()
        sentiment_df = sentiment_df.copy()

        price_df["Date"] = pd.to_datetime(price_df["Date"])
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

        if "ticker" in sentiment_df.columns:
            sentiment_df = sentiment_df[sentiment_df["ticker"] == ticker]

        daily_sentiment = sentiment_df.groupby("date").agg({
            "sentiment_positive": "mean",
            "sentiment_negative": "mean",
            "sentiment_neutral": "mean",
            "sentiment_score": "mean",
        }).reset_index()

        merged = pd.merge(
            price_df,
            daily_sentiment,
            left_on="Date",
            right_on="date",
            how="left",
        )

        sentiment_cols = [
            "sentiment_positive", "sentiment_negative",
            "sentiment_neutral", "sentiment_score",
        ]
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].ffill().fillna(0.0)

        if "date" in merged.columns and "Date" in merged.columns:
            merged = merged.drop(columns=["date"])

        logger.info(
            f"Merged features for {ticker}: "
            f"{len(merged)} rows, {len(merged.columns)} columns"
        )
        return merged

    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize feature columns using StandardScaler.

        Args:
            df: DataFrame with feature columns.
            feature_columns: Columns to normalize. Auto-detected if None.
            fit: If True, fits the scaler. Set False for test/inference data.

        Returns:
            DataFrame with normalized features.
        """
        df = df.copy()

        if feature_columns is None:
            feature_columns = self._get_feature_columns(df)

        existing_cols = [c for c in feature_columns if c in df.columns]

        if not existing_cols:
            logger.warning("No feature columns found to normalize")
            return df

        if fit:
            df[existing_cols] = self.scaler.fit_transform(
                df[existing_cols].fillna(0)
            )
            self._is_fitted = True
        else:
            if not self._is_fitted:
                raise RuntimeError("Scaler not fitted. Call with fit=True first.")
            df[existing_cols] = self.scaler.transform(
                df[existing_cols].fillna(0)
            )

        logger.info(f"Normalized {len(existing_cols)} feature columns")
        return df

    def prepare_final_dataset(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        ticker: str,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Full pipeline: indicators -> merge sentiment -> (optionally normalize) -> clean.

        Args:
            price_df: Raw OHLCV DataFrame.
            sentiment_df: Sentiment scores DataFrame.
            ticker: Ticker symbol.
            normalize: Whether to normalize features. Set False when building
                a multi-stock combined dataset (normalize after merging).

        Returns:
            Clean feature matrix ready for LSTM input.
        """
        df = self.add_technical_indicators(price_df)
        df = self.merge_sentiment_features(df, sentiment_df, ticker)

        df = df.dropna(subset=["rsi", "macd", "atr"]).reset_index(drop=True)

        if normalize:
            df = self.normalize_features(df)

        # Drop last row (no target available due to shift)
        df = df.iloc[:-1].reset_index(drop=True)

        logger.info(
            f"Final dataset for {ticker}: "
            f"{len(df)} rows, features: {self._get_feature_columns(df)}"
        )
        return df

    def _get_indicator_columns(self) -> list[str]:
        """Return list of technical indicator column names."""
        return [
            "rsi", "macd", "macd_signal_line", "macd_hist",
            "bb_upper", "bb_lower", "bb_pct",
            "obv", "atr",
            "returns", "log_returns", "volatility_20d",
        ]

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return all feature columns present in the DataFrame."""
        possible = self._get_indicator_columns() + [
            "sentiment_positive", "sentiment_negative",
            "sentiment_neutral", "sentiment_score",
        ]
        return [c for c in possible if c in df.columns]
