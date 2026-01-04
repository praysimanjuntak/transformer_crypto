#!/usr/bin/env python3
"""
Train decoder-only transformer for BTC return prediction.
Uses returns instead of raw prices to avoid distribution shift.

Usage:
    python train.py --data_dir 1h --epochs 100 --batch_size 512
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import Config
from src.dataset import create_dataloaders
from src.model import CandleTransformer
from src.train import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="btcusdt", help="Trading pair (btcusdt, xauusd)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--pred_horizon", type=int, default=4, help="Predict next N steps")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    base = Path("/home/pray/Documents/btc_transformer")
    data_dir = base / args.pair / args.timeframe
    checkpoint_dir = base / f"checkpoints_{args.pair}_{args.timeframe}"

    config = Config(
        d_model=128,
        n_heads=8,
        n_layers=2,
        d_ff=512,
        dropout=0.1,
        max_seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
    )

    print("=" * 50)
    print(f"OHLC Transformer - {args.pair.upper()} {args.timeframe}")
    print("=" * 50)
    print(f"d_model: {config.d_model}, heads: {config.n_heads}, layers: {config.n_layers}")
    print(f"features: {config.n_features} (OHLC), seq_len: {config.max_seq_len}")
    print(f"pred_horizon: {config.pred_horizon} steps")
    print(f"batch: {config.batch_size}, lr: {config.learning_rate}")
    print(f"data: {config.data_dir}")
    print("=" * 50)

    # Load data
    train_loader, test_loader, test_prices = create_dataloaders(config)

    # Save test prices for evaluation
    np.save(config.checkpoint_dir / "test_prices.npy", test_prices)

    # Create and train model
    model = CandleTransformer(config)
    trainer = Trainer(model, train_loader, test_loader, config)
    results = trainer.train()

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(results["train_losses"], label="Train")
    plt.plot(results["test_losses"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(config.checkpoint_dir / "losses.png")
    plt.close()

    print(f"\nDone! Checkpoint saved to {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
