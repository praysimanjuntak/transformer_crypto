#!/usr/bin/env python3
"""
Evaluate PatchTransformer forecasting model.

Usage:
    python predict_patch.py --pair btcusdt --timeframe 15m
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.config import PatchConfig
from src.model_patch import PatchTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="btcusdt", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_plots", type=int, default=5, help="Number of example plots")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, device, test_ohlc, seq_len, pred_len):
    """Evaluate model on test set."""
    model.eval()

    all_preds = []
    all_actuals = []
    all_naive = []

    n_samples = len(test_ohlc) - seq_len - pred_len

    for i in tqdm(range(0, n_samples, 64), desc="Evaluating"):
        batch_end = min(i + 64, n_samples)
        batch_x = []
        batch_y = []
        batch_naive = []

        for j in range(i, batch_end):
            x = test_ohlc[j : j + seq_len]
            y = test_ohlc[j + seq_len : j + seq_len + pred_len, 3]  # close prices
            naive = np.full(pred_len, x[-1, 3])  # last close price

            batch_x.append(x)
            batch_y.append(y)
            batch_naive.append(naive)

        x_tensor = torch.tensor(np.array(batch_x), dtype=torch.float32).to(device)
        preds = model(x_tensor).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_actuals.extend(batch_y)
        all_naive.extend(batch_naive)

    preds = np.array(all_preds)
    actuals = np.array(all_actuals)
    naive = np.array(all_naive)

    # Metrics
    model_mae = np.mean(np.abs(preds - actuals))
    naive_mae = np.mean(np.abs(naive - actuals))

    # Per-step metrics
    model_mae_per_step = np.mean(np.abs(preds - actuals), axis=0)
    naive_mae_per_step = np.mean(np.abs(naive - actuals), axis=0)

    # Direction accuracy
    pred_direction = preds[:, -1] > preds[:, 0]  # up over forecast horizon
    actual_direction = actuals[:, -1] > actuals[:, 0]
    direction_acc = np.mean(pred_direction == actual_direction)

    return {
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "model_mae_per_step": model_mae_per_step,
        "naive_mae_per_step": naive_mae_per_step,
        "direction_acc": direction_acc,
        "preds": preds,
        "actuals": actuals,
        "naive": naive,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base = Path("/home/pray/Documents/btc_transformer")
    checkpoint_dir = base / f"checkpoints_{args.pair}_{args.timeframe}_patch"
    checkpoint_path = checkpoint_dir / "model.pt"

    if not checkpoint_path.exists():
        print(f"Error: No checkpoint at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = PatchTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_ohlc = np.load(checkpoint_dir / "test_ohlc.npy")

    print(f"Test data: {len(test_ohlc):,} candles")
    print(f"Input: {config.max_seq_len} candles")
    print(f"Output: {config.pred_horizon} candles")

    results = evaluate(model, device, test_ohlc, config.max_seq_len, config.pred_horizon)

    ratio = results["model_mae"] / results["naive_mae"]

    print("\n" + "=" * 60)
    print(f"Forecasting Results ({config.pred_horizon} steps ahead)")
    print("=" * 60)
    print(f"Model MAE:     ${results['model_mae']:.2f}")
    print(f"Naive MAE:     ${results['naive_mae']:.2f}")
    print(f"Ratio:         {ratio:.3f}x {'(BETTER!)' if ratio < 1 else '(worse)'}")
    print(f"Direction Acc: {results['direction_acc']*100:.1f}%")
    print("=" * 60)

    # Plot MAE per step
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MAE per step
    steps = range(1, config.pred_horizon + 1)
    axes[0, 0].plot(steps, results["model_mae_per_step"], label="Model", linewidth=2)
    axes[0, 0].plot(steps, results["naive_mae_per_step"], label="Naive", linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel("Prediction Step")
    axes[0, 0].set_ylabel("MAE ($)")
    axes[0, 0].set_title("MAE per Prediction Step")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Ratio per step
    ratio_per_step = results["model_mae_per_step"] / results["naive_mae_per_step"]
    axes[0, 1].plot(steps, ratio_per_step, linewidth=2, color='green')
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label="Naive baseline")
    axes[0, 1].fill_between(steps, ratio_per_step, 1.0,
                            where=(ratio_per_step < 1), alpha=0.3, color='green', label="Better than naive")
    axes[0, 1].fill_between(steps, ratio_per_step, 1.0,
                            where=(ratio_per_step >= 1), alpha=0.3, color='red', label="Worse than naive")
    axes[0, 1].set_xlabel("Prediction Step")
    axes[0, 1].set_ylabel("Model/Naive Ratio")
    axes[0, 1].set_title("Relative Performance per Step")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Example predictions
    n_examples = min(args.num_plots, 3)
    for i in range(n_examples):
        idx = np.random.randint(0, len(results["preds"]))
        ax_idx = (1, i) if i < 2 else (1, 1)  # Adjust for 2x2 grid

        if i < 2:
            ax = axes[1, i]
            ax.plot(range(config.pred_horizon), results["actuals"][idx], label="Actual", linewidth=2)
            ax.plot(range(config.pred_horizon), results["preds"][idx], label="Predicted", linewidth=2)
            ax.plot(range(config.pred_horizon), results["naive"][idx], label="Naive", linestyle='--', alpha=0.7)
            ax.set_xlabel("Step")
            ax.set_ylabel("Price ($)")
            ax.set_title(f"Example Prediction #{i+1}")
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.savefig(checkpoint_dir / "evaluation.png")
    print(f"\nPlot saved to {checkpoint_dir / 'evaluation.png'}")

    # Verdict
    print("\n" + "=" * 60)
    if ratio < 0.95:
        print("VERDICT: Model beats naive baseline significantly!")
    elif ratio < 1.0:
        print("VERDICT: Model slightly better than naive")
    elif ratio < 1.05:
        print("VERDICT: Model roughly equal to naive")
    else:
        print("VERDICT: Model worse than naive")
    print("=" * 60)


if __name__ == "__main__":
    main()
