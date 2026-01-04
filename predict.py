#!/usr/bin/env python3
"""
Evaluate trained model with multi-step prediction.

Usage:
    python predict.py --data_dir 4h
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.dataset import load_data
from src.model import CandleTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="btcusdt", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="4h", help="Timeframe")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=500)
    return parser.parse_args()


def compute_ohlc_features(ohlc: np.ndarray) -> np.ndarray:
    """Compute OHLC returns relative to previous close."""
    prev_close = ohlc[:-1, 3]
    features = np.zeros((len(ohlc) - 1, 4), dtype=np.float32)
    features[:, 0] = (ohlc[1:, 0] - prev_close) / prev_close
    features[:, 1] = (ohlc[1:, 1] - prev_close) / prev_close
    features[:, 2] = (ohlc[1:, 2] - prev_close) / prev_close
    features[:, 3] = (ohlc[1:, 3] - prev_close) / prev_close
    return features


@torch.no_grad()
def evaluate(model, device, test_ohlc, seq_len, pred_horizon):
    """Evaluate model with multi-step prediction."""
    model.eval()

    features = compute_ohlc_features(test_ohlc)
    close_returns = features[:, 3]
    test_prices = test_ohlc[1:, 3]

    all_preds = []
    all_actuals = []
    all_last_prices = []

    n_samples = len(features) - seq_len - pred_horizon + 1

    for i in tqdm(range(0, n_samples, 64), desc="Evaluating"):
        batch_end = min(i + 64, n_samples)
        batch_x = []
        batch_y = []
        batch_prices = []

        for j in range(i, batch_end):
            x = features[j : j + seq_len]
            y = close_returns[j + seq_len : j + seq_len + pred_horizon]
            last_price = test_prices[j + seq_len - 1]

            batch_x.append(x)
            batch_y.append(y)
            batch_prices.append(last_price)

        x_tensor = torch.tensor(np.array(batch_x), dtype=torch.float32).to(device)

        # Model predicts all steps at once
        preds = model(x_tensor)  # (batch, pred_horizon)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_actuals.extend(batch_y)
        all_last_prices.extend(batch_prices)

    pred_returns = np.array(all_preds)
    actual_returns = np.array(all_actuals)
    last_prices = np.array(all_last_prices)

    # Convert returns to prices
    pred_prices = np.zeros_like(pred_returns)
    actual_prices = np.zeros_like(actual_returns)
    naive_prices = np.zeros_like(actual_returns)

    for step in range(pred_horizon):
        if step == 0:
            pred_prices[:, step] = last_prices * (1 + pred_returns[:, step])
            actual_prices[:, step] = last_prices * (1 + actual_returns[:, step])
        else:
            pred_prices[:, step] = pred_prices[:, step-1] * (1 + pred_returns[:, step])
            actual_prices[:, step] = actual_prices[:, step-1] * (1 + actual_returns[:, step])
        naive_prices[:, step] = last_prices

    # Metrics per step
    model_mae_per_step = np.mean(np.abs(pred_prices - actual_prices), axis=0)
    naive_mae_per_step = np.mean(np.abs(naive_prices - actual_prices), axis=0)
    return_mae_per_step = np.mean(np.abs(pred_returns - actual_returns), axis=0)

    # Direction accuracy per step
    pred_direction = pred_returns > 0
    actual_direction = actual_returns > 0
    direction_acc_per_step = np.mean(pred_direction == actual_direction, axis=0)

    return {
        "model_mae": model_mae_per_step,
        "naive_mae": naive_mae_per_step,
        "return_mae": return_mae_per_step,
        "direction_acc": direction_acc_per_step,
        "pred_prices": pred_prices,
        "actual_prices": actual_prices,
        "pred_returns": pred_returns,
        "actual_returns": actual_returns,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base = Path("/home/pray/Documents/btc_transformer")
    checkpoint_dir = base / f"checkpoints_{args.pair}_{args.timeframe}"
    checkpoint_path = checkpoint_dir / "model.pt"

    if not checkpoint_path.exists():
        print(f"Error: No checkpoint at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = CandleTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    data = load_data(config.data_dir)
    ohlc = data[["open", "high", "low", "close"]].values

    n = len(ohlc)
    train_end = int(n * config.train_ratio)
    test_ohlc = ohlc[train_end - 1:]

    print(f"Loaded {len(ohlc):,} candles, test: {len(test_ohlc):,}")
    print(f"Prediction horizon: {config.pred_horizon} steps")

    results = evaluate(model, device, test_ohlc, config.max_seq_len, config.pred_horizon)

    print("\n" + "=" * 70)
    print(f"Multi-Step Prediction ({config.pred_horizon} steps)")
    print("=" * 70)
    print(f"{'Step':<6} {'Return MAE':<12} {'Model $':<12} {'Naive $':<12} {'Ratio':<8} {'Dir Acc':<8}")
    print("-" * 70)
    for i in range(config.pred_horizon):
        ratio = results['model_mae'][i] / results['naive_mae'][i]
        print(f"{i+1:<6} {results['return_mae'][i]:<12.6f} ${results['model_mae'][i]:<11.2f} ${results['naive_mae'][i]:<11.2f} {ratio:<8.2f}x {results['direction_acc'][i]*100:<7.1f}%")
    print("=" * 70)
    print(f"Average direction accuracy: {np.mean(results['direction_acc'])*100:.1f}%")

    # Plot
    n = min(args.num_samples, len(results["pred_prices"]))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot step 1 predictions
    axes[0, 0].plot(results["actual_prices"][:n, 0], label="Actual", alpha=0.8)
    axes[0, 0].plot(results["pred_prices"][:n, 0], label="Predicted", alpha=0.8)
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Price (USD)")
    axes[0, 0].set_title("Step 1 Predictions")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot final step predictions
    step = config.pred_horizon - 1
    axes[0, 1].plot(results["actual_prices"][:n, step], label="Actual", alpha=0.8)
    axes[0, 1].plot(results["pred_prices"][:n, step], label="Predicted", alpha=0.8)
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Price (USD)")
    axes[0, 1].set_title(f"Step {config.pred_horizon} Predictions")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot MAE per step
    steps = range(1, config.pred_horizon + 1)
    axes[1, 0].plot(steps, results["model_mae"], 'o-', label="Model")
    axes[1, 0].plot(steps, results["naive_mae"], 's--', label="Naive")
    axes[1, 0].set_xlabel("Prediction Step")
    axes[1, 0].set_ylabel("MAE (USD)")
    axes[1, 0].set_title("MAE vs Prediction Horizon")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot direction accuracy per step
    axes[1, 1].bar(steps, results["direction_acc"] * 100)
    axes[1, 1].axhline(y=50, color='r', linestyle='--', label="Random")
    axes[1, 1].set_xlabel("Prediction Step")
    axes[1, 1].set_ylabel("Direction Accuracy (%)")
    axes[1, 1].set_title("Direction Accuracy per Step")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(checkpoint_dir / "predictions_multistep.png")
    print(f"Plot saved to {checkpoint_dir / 'predictions_multistep.png'}")


if __name__ == "__main__":
    main()
