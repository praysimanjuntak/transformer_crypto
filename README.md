# BTC Transformer: Can Transformers Predict Crypto Prices?

A systematic empirical study testing whether decoder-only transformers can predict short-term cryptocurrency and gold price movements.

**TL;DR:** After 10 experiments across different architectures, loss functions, assets, and timeframes, all approaches failed to beat the naive baseline of "predict no change." Short-term price movements appear to be fundamentally unpredictable.

## Results Summary

| Experiment | Approach | Result |
|------------|----------|--------|
| 1-2 | Raw price prediction | Distribution shift failure |
| 3 | Returns prediction (MSE) | Model predicts zero (1.00x naive) |
| 4 | OHLC features | No improvement (1.02x) |
| 5 | Multi-step loss | Still ~50% direction accuracy |
| 6 | Shorter timeframe (15m) | No improvement |
| 7 | Extended horizon (32 steps) | Predicts zero for all steps |
| 8 | Different asset (Gold) | Actually worse (1.10-1.33x) |
| 9 | Binary classification (BCE) | Collapsed to majority class |
| 10 | RevIN + Patching + MAE | Still 1.22x worse than naive |

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/btc-transformer.git
cd btc-transformer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Download Dataset

### Option 1: Binance Historical Data (BTCUSDT)

```bash
# Create data directory
mkdir -p btcusdt/15m

# Download script (downloads 2020-2025)
python download_data.py --pair BTCUSDT --timeframe 15m --start 2020-01 --end 2025-11
```

Or manually download from [Binance Data Vision](https://data.binance.vision/):

```bash
# Example: Download single month
wget https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2024-01.zip
unzip BTCUSDT-15m-2024-01.zip -d btcusdt/15m/

# Add headers if missing (Binance raw data has no headers)
# The download_data.py script handles this automatically
```

### Option 2: Manual Download

1. Go to https://data.binance.vision/
2. Navigate to: `data/spot/monthly/klines/BTCUSDT/15m/`
3. Download ZIP files for desired date range
4. Extract to `btcusdt/15m/` folder
5. Ensure CSV files have headers:
   ```
   open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore
   ```

### Data Format

Each CSV should have these columns:
```
open_time       - Unix timestamp in milliseconds
open            - Opening price
high            - Highest price
low             - Lowest price
close           - Closing price
volume          - Trading volume
close_time      - Candle close timestamp
quote_volume    - Quote asset volume
count           - Number of trades
taker_buy_volume - Taker buy base asset volume
taker_buy_quote_volume - Taker buy quote asset volume
ignore          - Ignore
```

## Usage

### Train Regression Model (Experiments 3-7)

```bash
# Basic training
python train.py --pair btcusdt --timeframe 15m --epochs 100

# With custom parameters
python train.py --pair btcusdt --timeframe 15m --epochs 100 --batch_size 128 --pred_horizon 4
```

### Train Classifier (Experiment 9)

```bash
python train_clf.py --pair btcusdt --timeframe 15m --horizon 32 --epochs 100
```

### Train PatchTransformer with RevIN (Experiment 10)

```bash
python train_patch.py --pair btcusdt --timeframe 15m --seq_len 336 --pred_len 96 --epochs 100
```

### Evaluate

```bash
# Regression
python predict.py --pair btcusdt --timeframe 15m

# Classifier
python predict_clf.py --pair btcusdt --timeframe 15m

# PatchTransformer
python predict_patch.py --pair btcusdt --timeframe 15m
```

## Project Structure

```
btc-transformer/
├── src/
│   ├── config.py          # Configuration dataclass
│   ├── dataset.py         # Returns-based dataset
│   ├── dataset_clf.py     # Classification dataset
│   ├── dataset_patch.py   # Forecasting dataset (raw prices)
│   ├── model.py           # Base transformer model
│   ├── model_clf.py       # Classification model
│   ├── model_patch.py     # PatchTransformer with RevIN
│   ├── train.py           # Trainer class (MSE)
│   ├── train_clf.py       # Classifier trainer (BCE)
│   └── train_patch.py     # Patch trainer (MAE)
├── train.py               # Main training script
├── train_clf.py           # Classification training
├── train_patch.py         # PatchTransformer training
├── predict.py             # Evaluation script
├── predict_clf.py         # Classification evaluation
├── predict_patch.py       # PatchTransformer evaluation
├── download_data.py       # Dataset download script
├── EXPERIMENTS.md         # Detailed experiment log
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Key Findings

1. **MSE loss → model predicts zero** (the mean)
2. **BCE loss → model predicts majority class**
3. **Modern techniques (RevIN, Patching, MAE) don't help**
4. **Different assets (BTC, Gold) → both unpredictable**
5. **Direction accuracy stuck at ~50%** (random)

## Conclusion

Short-term price movements appear to be fundamentally unpredictable using price data alone. This aligns with the Efficient Market Hypothesis. Potential extensions:
- External features (sentiment, on-chain data)
- Much longer timeframes (weekly/monthly)
- Volatility prediction instead of direction

## Citation

If you use this code in your research, please cite:

```
@misc{btc-transformer-2025,
  author = {Your Name},
  title = {BTC Transformer: Empirical Study on Price Prediction},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/btc-transformer}
}
```

## License

MIT License
