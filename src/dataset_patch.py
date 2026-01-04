"""
Dataset for time-series forecasting with raw OHLC values.
RevIN handles normalization, so we use raw prices here.

Input: 336 candles of OHLC
Output: 96 future close prices
"""

import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load all CSV files from the data directory and concatenate."""
    csv_files = sorted(glob.glob(str(data_dir / "*.csv")))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values("open_time").reset_index(drop=True)
    data = data.drop_duplicates(subset=["open_time"], keep="first")

    return data


class ForecastDataset(Dataset):
    """Dataset for time-series forecasting.

    Input: seq_len candles of OHLC (raw values)
    Output: pred_len future close prices (raw values)
    """

    def __init__(self, ohlc: np.ndarray, seq_len: int = 336, pred_len: int = 96):
        """
        Args:
            ohlc: (n, 4) array of [open, high, low, close]
            seq_len: input sequence length
            pred_len: prediction length
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.ohlc = ohlc.astype(np.float32)
        self.close = ohlc[:, 3].astype(np.float32)

    def __len__(self) -> int:
        return len(self.ohlc) - self.seq_len - self.pred_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: seq_len x 4 (OHLC raw values)
        x = self.ohlc[idx : idx + self.seq_len]

        # Target: next pred_len close prices (raw values)
        y = self.close[idx + self.seq_len : idx + self.seq_len + self.pred_len]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def create_dataloaders_patch(config: Config) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """Create train and test dataloaders for forecasting."""

    data = load_data(config.data_dir)
    ohlc = data[["open", "high", "low", "close"]].values

    print(f"Loaded {len(ohlc):,} candles from {config.data_dir}")

    # 70/30 split
    n = len(ohlc)
    train_end = int(n * config.train_ratio)

    train_ohlc = ohlc[:train_end]
    test_ohlc = ohlc[train_end:]

    print(f"Train: {len(train_ohlc):,} | Test: {len(test_ohlc):,}")

    # Create datasets
    train_dataset = ForecastDataset(
        train_ohlc, seq_len=config.max_seq_len, pred_len=config.pred_horizon
    )
    test_dataset = ForecastDataset(
        test_ohlc, seq_len=config.max_seq_len, pred_len=config.pred_horizon
    )

    print(f"Train samples: {len(train_dataset):,} | Test samples: {len(test_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader, test_ohlc
