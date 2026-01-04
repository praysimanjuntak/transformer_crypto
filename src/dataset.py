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


class CandleDataset(Dataset):
    """Dataset using OHLC features as returns relative to previous close."""

    def __init__(self, ohlc: np.ndarray, seq_len: int = 128, pred_horizon: int = 1):
        """
        Args:
            ohlc: (n, 4) array of [open, high, low, close]
            seq_len: sequence length
            pred_horizon: number of future steps to predict
        """
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        # Calculate OHLC returns relative to previous close
        # This preserves candle patterns while normalizing
        prev_close = ohlc[:-1, 3]  # previous close

        # Features: open, high, low, close returns relative to prev close
        self.features = np.zeros((len(ohlc) - 1, 4), dtype=np.float32)
        self.features[:, 0] = (ohlc[1:, 0] - prev_close) / prev_close  # open return (gap)
        self.features[:, 1] = (ohlc[1:, 1] - prev_close) / prev_close  # high return
        self.features[:, 2] = (ohlc[1:, 2] - prev_close) / prev_close  # low return
        self.features[:, 3] = (ohlc[1:, 3] - prev_close) / prev_close  # close return

        # Target is close return
        self.targets = self.features[:, 3].copy()

    def __len__(self) -> int:
        return len(self.features) - self.seq_len - self.pred_horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: seq_len x 4 features
        x = self.features[idx : idx + self.seq_len]
        # Target: next pred_horizon close returns
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.pred_horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """Create train and test dataloaders (70/30 split)."""

    data = load_data(config.data_dir)
    ohlc = data[["open", "high", "low", "close"]].values

    print(f"Loaded {len(ohlc):,} candles from {config.data_dir}")

    # 70/30 split
    n = len(ohlc)
    train_end = int(n * config.train_ratio)

    train_ohlc = ohlc[:train_end]
    test_ohlc = ohlc[train_end - 1:]  # overlap by 1 for continuity

    print(f"Train: {len(train_ohlc):,} | Test: {len(test_ohlc):,}")

    # Create datasets
    train_dataset = CandleDataset(train_ohlc, seq_len=config.max_seq_len, pred_horizon=config.pred_horizon)
    test_dataset = CandleDataset(test_ohlc, seq_len=config.max_seq_len, pred_horizon=config.pred_horizon)

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

    # Return test close prices for converting returns back to prices
    test_prices = test_ohlc[:, 3]  # close prices
    return train_loader, test_loader, test_prices
