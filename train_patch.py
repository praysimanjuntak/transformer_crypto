#!/usr/bin/env python3
"""
Train PatchTransformer with RevIN for time-series forecasting.

Features:
- RevIN (Reversible Instance Normalization)
- Patching (group candles into patches)
- MAE (L1) loss
- AdamW + Cosine Annealing

Usage:
    python train_patch.py --pair btcusdt --timeframe 15m --seq_len 336 --pred_len 96
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import PatchConfig
from src.dataset_patch import create_dataloaders_patch
from src.model_patch import PatchTransformer
from src.train_patch import PatchTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="btcusdt", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe")
    parser.add_argument("--seq_len", type=int, default=336, help="Input sequence length")
    parser.add_argument("--pred_len", type=int, default=96, help="Prediction length")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    base = Path("/home/pray/Documents/btc_transformer")
    data_dir = base / args.pair / args.timeframe
    checkpoint_dir = base / f"checkpoints_{args.pair}_{args.timeframe}_patch"

    config = PatchConfig(
        d_model=128,
        n_heads=8,
        n_layers=3,  # Slightly deeper for longer sequences
        d_ff=512,
        dropout=0.1,
        max_seq_len=args.seq_len,
        pred_horizon=args.pred_len,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
    )

    # Calculate derived values
    num_patches = args.seq_len // args.patch_size
    hours_input = args.seq_len * 15 / 60  # for 15m timeframe
    hours_output = args.pred_len * 15 / 60

    print("=" * 60)
    print(f"PatchTransformer - {args.pair.upper()} {args.timeframe}")
    print("=" * 60)
    print(f"Input:  {args.seq_len} candles ({hours_input:.1f}h) → {num_patches} patches")
    print(f"Output: {args.pred_len} candles ({hours_output:.1f}h)")
    print(f"Patch size: {args.patch_size}")
    print(f"Architecture: d={config.d_model}, heads={config.n_heads}, layers={config.n_layers}")
    print(f"Loss: MAE (L1)")
    print("=" * 60)

    # Load data
    train_loader, test_loader, test_ohlc = create_dataloaders_patch(config)

    # Save test data for evaluation
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.checkpoint_dir / "test_ohlc.npy", test_ohlc)

    # Create and train model
    model = PatchTransformer(config)
    trainer = PatchTrainer(model, train_loader, test_loader, config)
    results = trainer.train()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(results["train_losses"], label="Train")
    axes[0].plot(results["test_losses"], label="Test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # MAE plot
    axes[1].plot(results["test_maes"], label="Test MAE ($)")
    axes[1].axhline(y=results["best_mae"], color='g', linestyle='--', label=f"Best: ${results['best_mae']:.2f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE ($)")
    axes[1].set_title("Test MAE over Training")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(config.checkpoint_dir / "training.png")
    plt.close()

    print(f"\nDone! Best MAE: ${results['best_mae']:.2f}")
    print(f"Checkpoint saved to {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
