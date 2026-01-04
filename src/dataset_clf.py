"""
Dataset for trend classification.
Target: Will price be higher after N steps? (binary)
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


class TrendDataset(Dataset):
    """Dataset for trend classification.

    Target: 1 if price after `horizon` steps > current price, else 0
    """

    def __init__(self, ohlc: np.ndarray, seq_len: int = 128, horizon: int = 32):
        """
        Args:
            ohlc: (n, 4) array of [open, high, low, close]
            seq_len: input sequence length
            horizon: how many steps ahead to predict trend
        """
        self.seq_len = seq_len
        self.horizon = horizon

        # Calculate OHLC returns relative to previous close
        prev_close = ohlc[:-1, 3]
        self.features = np.zeros((len(ohlc) - 1, 4), dtype=np.float32)
        self.features[:, 0] = (ohlc[1:, 0] - prev_close) / prev_close
        self.features[:, 1] = (ohlc[1:, 1] - prev_close) / prev_close
        self.features[:, 2] = (ohlc[1:, 2] - prev_close) / prev_close
        self.features[:, 3] = (ohlc[1:, 3] - prev_close) / prev_close

        # Close prices for computing trend labels
        self.close_prices = ohlc[1:, 3]

    def __len__(self) -> int:
        return len(self.features) - self.seq_len - self.horizon

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: seq_len x 4 features
        x = self.features[idx : idx + self.seq_len]

        # Target: is price at (idx + seq_len + horizon) > price at (idx + seq_len)?
        current_price = self.close_prices[idx + self.seq_len - 1]
        future_price = self.close_prices[idx + self.seq_len + self.horizon - 1]

        # Binary label: 1 if up, 0 if down
        label = 1.0 if future_price > current_price else 0.0

        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def create_dataloaders_clf(config: Config) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """Create train and test dataloaders for classification."""

    data = load_data(config.data_dir)
    ohlc = data[["open", "high", "low", "close"]].values

    print(f"Loaded {len(ohlc):,} candles from {config.data_dir}")

    # 70/30 split
    n = len(ohlc)
    train_end = int(n * config.train_ratio)

    train_ohlc = ohlc[:train_end]
    test_ohlc = ohlc[train_end - 1:]

    print(f"Train: {len(train_ohlc):,} | Test: {len(test_ohlc):,}")

    # Create datasets
    train_dataset = TrendDataset(train_ohlc, seq_len=config.max_seq_len, horizon=config.pred_horizon)
    test_dataset = TrendDataset(test_ohlc, seq_len=config.max_seq_len, horizon=config.pred_horizon)

    # Count class balance
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    up_pct = sum(train_labels) / len(train_labels) * 100
    print(f"Train class balance: {up_pct:.1f}% up, {100-up_pct:.1f}% down")

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

    test_prices = test_ohlc[:, 3]
    return train_loader, test_loader, test_prices
