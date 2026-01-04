#!/usr/bin/env python3
"""
Evaluate trend classifier.

Usage:
    python predict_clf.py --pair btcusdt --timeframe 15m
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.dataset_clf import load_data
from src.model_clf import TrendClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="btcusdt", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe")
    parser.add_argument("--device", type=str, default="cuda")
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
def evaluate(model, device, test_ohlc, seq_len, horizon):
    """Evaluate classifier on test set."""
    model.eval()

    features = compute_ohlc_features(test_ohlc)
    close_prices = test_ohlc[1:, 3]

    all_probs = []
    all_preds = []
    all_labels = []
    all_returns = []

    n_samples = len(features) - seq_len - horizon

    for i in tqdm(range(0, n_samples, 64), desc="Evaluating"):
        batch_end = min(i + 64, n_samples)
        batch_x = []
        batch_labels = []
        batch_returns = []

        for j in range(i, batch_end):
            x = features[j : j + seq_len]
            current_price = close_prices[j + seq_len - 1]
            future_price = close_prices[j + seq_len + horizon - 1]

            label = 1.0 if future_price > current_price else 0.0
            ret = (future_price - current_price) / current_price

            batch_x.append(x)
            batch_labels.append(label)
            batch_returns.append(ret)

        x_tensor = torch.tensor(np.array(batch_x), dtype=torch.float32).to(device)
        logits = model(x_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(batch_labels)
        all_returns.extend(batch_returns)

    probs = np.array(all_probs)
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    returns = np.array(all_returns)

    # Metrics
    accuracy = np.mean(preds == labels)

    # Separate accuracy for up and down
    up_mask = labels == 1
    down_mask = labels == 0
    up_acc = np.mean(preds[up_mask] == labels[up_mask]) if up_mask.sum() > 0 else 0
    down_acc = np.mean(preds[down_mask] == labels[down_mask]) if down_mask.sum() > 0 else 0

    # Confidence analysis
    high_conf_mask = (probs > 0.6) | (probs < 0.4)
    high_conf_acc = np.mean(preds[high_conf_mask] == labels[high_conf_mask]) if high_conf_mask.sum() > 0 else 0

    # Simulated trading: go long when predict up, short when predict down
    positions = np.where(preds == 1, 1, -1)
    trade_returns = positions * returns
    total_return = np.sum(trade_returns)
    win_rate = np.mean(trade_returns > 0)

    return {
        "accuracy": accuracy,
        "up_accuracy": up_acc,
        "down_accuracy": down_acc,
        "high_conf_accuracy": high_conf_acc,
        "high_conf_pct": high_conf_mask.mean(),
        "class_balance": labels.mean(),
        "total_return": total_return,
        "win_rate": win_rate,
        "probs": probs,
        "preds": preds,
        "labels": labels,
        "returns": returns,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base = Path("/home/pray/Documents/btc_transformer")
    checkpoint_dir = base / f"checkpoints_{args.pair}_{args.timeframe}_clf"
    checkpoint_path = checkpoint_dir / "model.pt"

    if not checkpoint_path.exists():
        print(f"Error: No checkpoint at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = TrendClassifier(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    data = load_data(config.data_dir)
    ohlc = data[["open", "high", "low", "close"]].values

    n = len(ohlc)
    train_end = int(n * config.train_ratio)
    test_ohlc = ohlc[train_end - 1:]

    print(f"Loaded {len(ohlc):,} candles, test: {len(test_ohlc):,}")
    print(f"Horizon: {config.pred_horizon} steps")

    results = evaluate(model, device, test_ohlc, config.max_seq_len, config.pred_horizon)

    print("\n" + "=" * 60)
    print(f"Trend Classification Results ({config.pred_horizon} steps)")
    print("=" * 60)
    print(f"Overall Accuracy:     {results['accuracy']*100:.2f}%")
    print(f"  - Up predictions:   {results['up_accuracy']*100:.2f}%")
    print(f"  - Down predictions: {results['down_accuracy']*100:.2f}%")
    print(f"High Confidence Acc:  {results['high_conf_accuracy']*100:.2f}% ({results['high_conf_pct']*100:.1f}% of samples)")
    print("-" * 60)
    print(f"Class Balance:        {results['class_balance']*100:.1f}% up / {(1-results['class_balance'])*100:.1f}% down")
    print("-" * 60)
    print(f"Simulated Trading:")
    print(f"  - Win Rate:         {results['win_rate']*100:.2f}%")
    print(f"  - Total Return:     {results['total_return']*100:.2f}%")
    print("=" * 60)

    # Verdict
    if results['accuracy'] > 0.52:
        print("VERDICT: Model shows potential edge (>52% accuracy)")
    elif results['accuracy'] > 0.50:
        print("VERDICT: Marginal edge, likely noise")
    else:
        print("VERDICT: No edge, random or worse")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Probability distribution
    axes[0, 0].hist(results['probs'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=0.5, color='r', linestyle='--', label="Threshold")
    axes[0, 0].set_xlabel("Predicted Probability (Up)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Prediction Probability Distribution")
    axes[0, 0].legend()

    # Accuracy by confidence
    conf_bins = np.linspace(0, 1, 11)
    conf_accs = []
    conf_counts = []
    for i in range(len(conf_bins) - 1):
        mask = (results['probs'] >= conf_bins[i]) & (results['probs'] < conf_bins[i+1])
        if mask.sum() > 0:
            conf_accs.append(np.mean(results['preds'][mask] == results['labels'][mask]))
            conf_counts.append(mask.sum())
        else:
            conf_accs.append(0)
            conf_counts.append(0)

    x_pos = (conf_bins[:-1] + conf_bins[1:]) / 2
    axes[0, 1].bar(x_pos, conf_accs, width=0.08, edgecolor='black', alpha=0.7)
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label="Random")
    axes[0, 1].set_xlabel("Confidence Level")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Accuracy by Confidence Level")
    axes[0, 1].legend()

    # Cumulative returns (trading simulation)
    cum_returns = np.cumsum(results['returns'] * np.where(results['preds'] == 1, 1, -1))
    axes[1, 0].plot(cum_returns * 100)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel("Trade #")
    axes[1, 0].set_ylabel("Cumulative Return (%)")
    axes[1, 0].set_title("Simulated Trading Performance")
    axes[1, 0].grid(True)

    # Confusion matrix style
    tp = np.sum((results['preds'] == 1) & (results['labels'] == 1))
    fp = np.sum((results['preds'] == 1) & (results['labels'] == 0))
    tn = np.sum((results['preds'] == 0) & (results['labels'] == 0))
    fn = np.sum((results['preds'] == 0) & (results['labels'] == 1))

    conf_matrix = np.array([[tn, fp], [fn, tp]])
    axes[1, 1].imshow(conf_matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, f"{conf_matrix[i, j]}", ha='center', va='center', fontsize=14)
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['Pred Down', 'Pred Up'])
    axes[1, 1].set_yticklabels(['Actual Down', 'Actual Up'])
    axes[1, 1].set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(checkpoint_dir / "evaluation.png")
    print(f"\nPlot saved to {checkpoint_dir / 'evaluation.png'}")


if __name__ == "__main__":
    main()
