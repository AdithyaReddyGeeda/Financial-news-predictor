"""
PyTorch Dataset for Time-Series Stock Prediction.

Creates sliding-window sequences from the feature matrix
for LSTM training and inference.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yaml

logger = logging.getLogger(__name__)

# Feature columns used as LSTM input
DEFAULT_FEATURE_COLS = [
    "rsi", "macd", "macd_signal_line", "macd_hist",
    "bb_pct", "obv", "atr",
    "returns", "log_returns", "volatility_20d",
    "sentiment_positive", "sentiment_negative",
    "sentiment_neutral", "sentiment_score",
]


class StockSequenceDataset(Dataset):
    """
    Sliding-window dataset for stock price prediction.

    Each sample is a (sequence, target) pair where:
    - sequence: (seq_len, num_features) tensor of historical features
    - target: dict with 'direction' (0/1) and 'return' (float)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 30,
        feature_columns: Optional[list[str]] = None,
        target_direction_col: str = "target_direction",
        target_return_col: str = "target_return",
    ):
        """
        Args:
            df: Feature DataFrame (already normalized, NaN-free).
            seq_len: Length of each input sequence (trading days).
            feature_columns: Columns to use as features. Auto-detected if None.
            target_direction_col: Column name for direction target (0/1).
            target_return_col: Column name for return target (float).
        """
        self.seq_len = seq_len

        if feature_columns is None:
            feature_columns = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]

        if not feature_columns:
            raise ValueError("No feature columns found in DataFrame")

        self.feature_names = feature_columns
        self.num_features = len(feature_columns)

        features = df[feature_columns].values.astype(np.float32)
        self.features = torch.FloatTensor(features)

        if target_direction_col in df.columns:
            self.directions = torch.FloatTensor(
                df[target_direction_col].values.astype(np.float32)
            )
        else:
            self.directions = torch.zeros(len(df))

        if target_return_col in df.columns:
            self.returns = torch.FloatTensor(
                df[target_return_col].values.astype(np.float32)
            )
        else:
            self.returns = torch.zeros(len(df))

        self.valid_indices = len(df) - seq_len

        logger.info(
            f"Dataset created: {self.valid_indices} samples, "
            f"seq_len={seq_len}, features={self.num_features} "
            f"({feature_columns})"
        )

    def __len__(self) -> int:
        return max(0, self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            sequence: (seq_len, num_features) tensor
            targets: dict with 'direction' and 'return' tensors
        """
        end_idx = idx + self.seq_len
        sequence = self.features[idx:end_idx]

        targets = {
            "direction": self.directions[end_idx],
            "return": self.returns[end_idx],
        }

        return sequence, targets


def create_data_splits(
    df: pd.DataFrame,
    config_path: str = "config.yaml",
    feature_columns: Optional[list[str]] = None,
) -> tuple[StockSequenceDataset, StockSequenceDataset, StockSequenceDataset]:
    """
    Create chronological train/val/test splits.

    IMPORTANT: No shuffling — time-series data must maintain order
    to avoid lookahead bias.

    Args:
        df: Full feature DataFrame.
        config_path: Path to config file.
        feature_columns: Feature column names.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    seq_len = model_config.get("seq_len", 30)
    train_ratio = model_config.get("train_split", 0.70)
    val_ratio = model_config.get("val_split", 0.15)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    logger.info(
        f"Data splits — Train: {len(train_df)}, "
        f"Val: {len(val_df)}, Test: {len(test_df)}"
    )

    train_ds = StockSequenceDataset(train_df, seq_len, feature_columns)
    val_ds = StockSequenceDataset(val_df, seq_len, feature_columns)
    test_ds = StockSequenceDataset(test_df, seq_len, feature_columns)

    return train_ds, val_ds, test_ds
