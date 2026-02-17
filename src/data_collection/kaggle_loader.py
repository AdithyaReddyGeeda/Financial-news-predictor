"""
Kaggle Financial News Dataset Loader.

Loads and parses CSV datasets (e.g., "Financial News Headlines" from Kaggle)
into a standardized DataFrame format for downstream processing.

Expected CSV columns (flexible - auto-detects common column names):
  - headline / title / text
  - date / publish_date / timestamp
  - source (optional)
  - ticker / stock (optional)
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Common column name mappings for auto-detection
HEADLINE_COLS = ["headline", "title", "text", "news", "description", "content"]
DATE_COLS = ["date", "publish_date", "timestamp", "published", "datetime", "time"]
TICKER_COLS = ["ticker", "stock", "symbol", "company"]
SOURCE_COLS = ["source", "publisher", "provider"]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find the first matching column name from a list of candidates."""
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    return None


class KaggleNewsLoader:
    """Load and standardize Kaggle financial news CSV datasets."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.dataset_path = self.config["data"].get(
            "kaggle_dataset_path", "data/raw/financial_news.csv"
        )

    def load(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load a financial news CSV and return a standardized DataFrame.

        Args:
            filepath: Path to CSV file. Defaults to config path.

        Returns:
            DataFrame with columns: [date, headline, ticker, source]
        """
        path = filepath or self.dataset_path

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found at '{path}'. "
                f"Download from Kaggle and place it in data/raw/."
            )

        logger.info(f"Loading Kaggle dataset from {path}")

        df = pd.read_csv(path, low_memory=False, encoding="latin-1")

        # Detect headerless CSVs (e.g., Ankurzing financial news dataset)
        # If column names look like data values (sentiment labels), treat as headerless
        cols_lower = [str(c).lower().strip() for c in df.columns]
        sentiment_labels = {"positive", "negative", "neutral"}
        if any(c in sentiment_labels for c in cols_lower):
            logger.info("Detected headerless CSV — re-reading with header=None")
            df = pd.read_csv(path, header=None, low_memory=False, encoding="latin-1")
            if df.shape[1] == 2:
                df.columns = ["sentiment", "headline"]
            elif df.shape[1] >= 2:
                df.columns = [f"col_{i}" if i > 1 else ["sentiment", "headline"][i]
                              for i in range(df.shape[1])]

        logger.info(f"Raw dataset: {len(df)} rows, columns: {list(df.columns)}")

        df = self._standardize_columns(df)
        df = self._clean(df)

        logger.info(f"Standardized dataset: {len(df)} rows")
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map detected columns to standard names."""
        result = pd.DataFrame()

        headline_col = _find_column(df, HEADLINE_COLS)
        if headline_col is None:
            raise ValueError(
                f"Could not find a headline column. "
                f"Expected one of: {HEADLINE_COLS}. "
                f"Found: {list(df.columns)}"
            )
        result["headline"] = df[headline_col].astype(str)

        date_col = _find_column(df, DATE_COLS)
        if date_col:
            result["date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            logger.warning("No date column found. Using index as date proxy.")
            result["date"] = pd.NaT

        ticker_col = _find_column(df, TICKER_COLS)
        result["ticker"] = df[ticker_col].astype(str) if ticker_col else "UNKNOWN"

        source_col = _find_column(df, SOURCE_COLS)
        result["source"] = df[source_col].astype(str) if source_col else "kaggle"

        return result

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: drop nulls, filter short headlines, sort by date."""
        min_len = self.config["preprocessing"].get("min_headline_length", 10)

        df = df.dropna(subset=["headline"])
        df = df[df["headline"].str.len() >= min_len]
        df = df.drop_duplicates(subset=["headline"])

        if df["date"].notna().any():
            df = df.sort_values("date").reset_index(drop=True)

        return df

    def load_multiple(self, filepaths: list[str]) -> pd.DataFrame:
        """Load and concatenate multiple CSV files."""
        dfs = []
        for fp in filepaths:
            try:
                dfs.append(self.load(fp))
            except Exception as e:
                logger.error(f"Failed to load {fp}: {e}")
                continue

        if not dfs:
            raise ValueError("No datasets were successfully loaded.")

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["headline"]).reset_index(drop=True)

        logger.info(f"Combined dataset: {len(combined)} rows from {len(dfs)} files")
        return combined
