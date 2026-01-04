#!/usr/bin/env python3
"""
Train trend classifier.

Target: Will price be higher after N steps? (binary classification)

Usage:
    python train_clf.py --pair btcusdt --timeframe 15m --horizon 32 --epochs 100
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.dataset_clf import create_dataloaders_clf
from src.model_clf import TrendClassifier
from src.train_clf import ClassifierTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="btcusdt", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe")
    parser.add_argument("--horizon", type=int, default=32, help="Prediction horizon (steps)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    base = Path("/home/pray/Documents/btc_transformer")
    data_dir = base / args.pair / args.timeframe
    checkpoint_dir = base / f"checkpoints_{args.pair}_{args.timeframe}_clf"

    config = Config(
        d_model=128,
        n_heads=8,
        n_layers=2,
        d_ff=512,
        dropout=0.1,
        max_seq_len=args.seq_len,
        pred_horizon=args.horizon,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
    )

    print("=" * 60)
    print(f"Trend Classifier - {args.pair.upper()} {args.timeframe}")
    print("=" * 60)
    print(f"Question: Will price be higher in {args.horizon} steps?")
    print(f"d_model: {config.d_model}, heads: {config.n_heads}, layers: {config.n_layers}")
    print(f"seq_len: {config.max_seq_len}, horizon: {config.pred_horizon}")
    print(f"batch: {config.batch_size}, lr: {config.learning_rate}")
    print("=" * 60)

    # Load data
    train_loader, test_loader, test_prices = create_dataloaders_clf(config)

    # Save test prices for evaluation
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.checkpoint_dir / "test_prices.npy", test_prices)

    # Create and train model
    model = TrendClassifier(config)
    trainer = ClassifierTrainer(model, train_loader, test_loader, config)
    results = trainer.train()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(results["train_losses"], label="Train")
    axes[0].plot(results["test_losses"], label="Test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot([acc * 100 for acc in results["test_accs"]], label="Test Accuracy")
    axes[1].axhline(y=50, color='r', linestyle='--', label="Random (50%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"Trend Prediction Accuracy (Best: {results['best_acc']*100:.2f}%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(config.checkpoint_dir / "training.png")
    plt.close()

    print(f"\nDone! Best accuracy: {results['best_acc']*100:.2f}%")
    print(f"Checkpoint saved to {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
