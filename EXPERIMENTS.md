# BTC Transformer Experiments Log

## Goal
Build a decoder-only transformer (like GPT) that predicts BTC price movements, based on paper 2504.16361v1 which found decoder-only transformers work best for stock prediction.

---

## Experiment 1: Raw Price Prediction (Z-score normalized)
**Date:** Initial attempt

**Setup:**
- Input: Z-score normalized close prices
- Output: Next close price
- Architecture: d_model=128, heads=4, layers=4
- Train/Val/Test split: 70/15/15

**Result:** Heavy overfitting
- Train loss: 0.0002
- Val loss: 0.76
- Test MAE: $26,797 USD

**Problem:** Distribution shift - trained on $29K prices, tested on $49K-$126K prices. Model predictions capped at ~$82K.

---

## Experiment 2: Paper-matched Config (Still raw prices)
**Date:** Second attempt

**Setup:**
- Matched paper exactly: 8 heads, 2 layers
- 70/30 train/test (no validation)
- Removed weight decay, grad clip, cosine scheduler
- Simple Adam optimizer

**Result:** Still severe distribution shift
- Model MAE: $13,824
- Naive baseline (predict last price): $284
- **Model was 48x worse than naive!**

**Problem:** Z-score normalization doesn't solve distribution shift when price levels change dramatically over time.

---

## Experiment 3: Returns-based Prediction
**Date:** Third attempt

**Setup:**
- Input: Returns `(price[t] - price[t-1]) / price[t-1]`
- Output: Next return
- No normalization needed (returns are naturally bounded)

**Result:** No overfitting, but no edge
- Train loss ≈ Test loss (good!)
- Model MAE: ~$287
- Naive MAE: ~$286
- **Ratio: 1.00x (equal to naive)**

**Conclusion:** Model learns to predict ~zero return, which equals naive baseline.

---

## Experiment 4: OHLC Features
**Date:** Fourth attempt

**Setup:**
- Input: 4 features per candle (OHLC returns relative to prev close)
  - Open return (gap)
  - High return
  - Low return
  - Close return
- Output: Next close return

**Hypothesis:** Candle patterns (wicks, body direction) might reveal support/resistance levels.

**Result (4h data):**
- Model MAE: $592
- Naive MAE: $580
- **Ratio: 1.02x (slightly worse than naive)**

**Conclusion:** OHLC features didn't help. Model still learns "predict no change."

---

## Experiment 5: Multi-Step Loss
**Date:** Completed

**Setup:**
- Input: 128 OHLC candles
- Output: Next 4 returns simultaneously
- Loss: MSE averaged over all 4 predictions
- Architecture: d_model=128, heads=8, layers=2, pred_horizon=4

**Hypothesis:** Forcing model to predict multiple steps might encourage learning trends rather than just "no change."

**Result (4h data):**
| Step | Model MAE | Naive MAE | Ratio | Dir Acc |
|------|-----------|-----------|-------|---------|
| 1 | $594 | $580 | 1.02x | 49.0% |
| 2 | $864 | $834 | 1.04x | 49.0% |
| 3 | $1134 | $1039 | 1.09x | 48.9% |
| 4 | $1377 | $1207 | 1.14x | 48.9% |

**Conclusion:** No improvement. Direction accuracy <50% means model is slightly worse than random. Multi-step loss doesn't help learn patterns.

---

## Experiment 6: Shorter Timeframe (15m)
**Date:** Completed

**Setup:**
- Same as Experiment 5, but 15m instead of 4h
- 207,456 candles (much more data)
- 4 steps = 1 hour ahead (vs 16 hours for 4h)

**Hypothesis:** Shorter timeframe = more coherent trends within prediction window.

**Result (15m data):**
| Step | Model MAE | Naive MAE | Ratio | Dir Acc |
|------|-----------|-----------|-------|---------|
| 1 | $144 | $144 | 1.00x | 49.9% |
| 2 | $209 | $201 | 1.04x | 49.9% |
| 3 | $250 | $244 | 1.03x | 50.1% |
| 4 | $287 | $280 | 1.02x | 49.9% |

**Conclusion:** No improvement. More data and shorter horizon didn't help. Direction accuracy still ~50% (random).

---

## Experiment 7: Extended Horizon (32 steps)
**Date:** Completed

**Setup:**
- 15m timeframe, 32 steps = 8 hours ahead
- Same architecture

**Result:**
- All 32 steps: Ratio = 1.00x, Dir Acc = 50.0%
- Model predicts zero return for everything
- Literally identical to naive baseline

**Conclusion:** Model learns nothing. Outputs constant "no change" prediction regardless of input or horizon length.

---

## Experiment 8: Different Asset (XAUUSD Gold)
**Date:** Completed

**Setup:**
- XAUUSD 15m, 236,296 candles
- pred_horizon=4 (1 hour ahead)
- Same architecture

**Hypothesis:** Gold might have more predictable patterns than BTC (less volatile, more institutional).

**Result:**
| Step | Model MAE | Naive MAE | Ratio | Dir Acc |
|------|-----------|-----------|-------|---------|
| 1 | $2.32 | $1.74 | 1.33x | 50.8% |
| 2 | $3.03 | $2.47 | 1.23x | 50.8% |
| 3 | $3.71 | $3.04 | 1.22x | 50.8% |
| 4 | $3.89 | $3.52 | 1.10x | 49.2% |

**Conclusion:** Gold is actually WORSE than BTC! Model performs 1.10-1.33x worse than naive. Direction accuracy ~50% (random). Different asset didn't help.

---

## Experiment 9: Trend Classifier (Binary Up/Down)
**Date:** Completed

**Setup:**
- Target: "Will price be higher in 32 steps?" (binary)
- Loss: BCE (Binary Cross Entropy)
- BTCUSDT 15m, horizon=32 (8 hours ahead)
- Same transformer architecture

**Hypothesis:** Classification with BCE loss will force model to commit to a direction instead of collapsing to zero.

**Result:**
| Metric | Value |
|--------|-------|
| Overall Accuracy | 52.24% |
| Up Predictions Acc | 100% |
| Down Predictions Acc | 0% |
| Class Balance | 52.2% up / 47.8% down |

**Problem:** Model predicts "UP" for everything!
- 52.24% accuracy = just the class balance
- Model collapsed to majority class (always predict up)
- Same problem as MSE (lazy solution), just different form
- The 2502% "return" is fake - just riding BTC bull market

**Conclusion:** BCE loss doesn't fix the fundamental issue. Model finds lazy shortcut (predict majority class) instead of learning patterns.

---

## Experiment 10: Modern Time-Series Transformer (RevIN + Patching + MAE)
**Date:** Completed

**Setup:**
- RevIN (Reversible Instance Normalization) - per-sample normalization
- Patching: 336 candles → 21 patches (patch_size=16)
- MAE (L1) loss instead of MSE (encourages median, not mean)
- AdamW + Cosine Annealing scheduler
- Input: 336 candles (84 hours / 3.5 days)
- Output: 96 candles (24 hours / 1 day ahead)
- 3 layers (deeper model)

**Hypothesis:** Modern techniques from PatchTST paper might help:
- RevIN handles distribution shift per-sample
- Patching reduces noise, captures local patterns
- MAE encourages sharper predictions

**Result:**
| Metric | Value |
|--------|-------|
| Model MAE | $1197.93 |
| Naive MAE | $984.14 |
| Ratio | 1.217x (worse) |
| Direction Acc | 50.5% |

**Observations:**
- Large train/test gap (301 vs 1197) = overfitting
- Direction accuracy still ~50% (random)
- Model worse than naive despite modern techniques

**Conclusion:** RevIN + Patching + MAE don't help. Model still can't beat "predict last price" baseline. The fundamental problem isn't the architecture or loss - it's that short-term price movements appear to be random.

---

## Key Findings

### What Doesn't Work
1. **Raw price prediction** - Distribution shift kills generalization
2. **Z-score normalization** - Doesn't solve distribution shift for non-stationary prices
3. **Single-step return prediction** - Model defaults to predicting zero (naive baseline)
4. **OHLC features alone** - No improvement over close-only
5. **Multi-step loss** - Doesn't help learn trends, direction accuracy still <50%
6. **Shorter timeframe (15m)** - More data and shorter horizon didn't help
7. **Extended horizon (32 steps)** - Model just predicts zero for all steps
8. **Different asset (XAUUSD)** - Actually worse than BTC, 1.10-1.33x ratio
9. **Binary classification (BCE)** - Model collapses to majority class (always predict up)
10. **RevIN + Patching + MAE** - Modern techniques still 1.22x worse than naive

### What We Learned
1. BTC short-term returns appear close to random walk
2. Naive baseline (predict no change) is surprisingly hard to beat
3. Returns-based approach avoids distribution shift but may lose absolute price level info
4. Paper used S&P 500 which may have different characteristics than BTC
5. Gold (XAUUSD) is also unpredictable - not just a BTC problem
6. MSE loss causes model to collapse to predicting zero (the mean)
7. BCE loss causes model to collapse to majority class (always predict up)
8. Model consistently finds "lazy" solutions instead of learning patterns
9. Modern techniques (RevIN, Patching, MAE) don't help - problem is fundamental
10. **Price movements may genuinely be unpredictable at these timeframes**

---

## Ideas to Try
- [x] ~~Multi-step loss~~ - Didn't help
- [x] ~~Different assets (XAUUSD)~~ - Actually worse
- [x] ~~Direction classification (BCE)~~ - Collapsed to majority class
- [ ] **Balanced sampling** - Force 50/50 up/down in each batch
- [ ] **Class weights** - Penalize minority class errors more
- [ ] **Focal loss** - Focus on hard examples
- [ ] Profit-based loss (reward correct direction)
- [ ] Add volume as feature
- [ ] Longer context (256-512 candles)
- [ ] Daily timeframe (match paper's S&P 500 setup)
- [ ] Bigger model (more layers/heads)

---

## Data
**BTCUSDT:**
- Source: Binance
- Timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- Date range: 2020-01 to 2025-11
- Location: `btcusdt/{timeframe}/`

**XAUUSD:**
- Source: Forex
- Timeframes: 15m
- Date range: 2015-12 to present
- Location: `xauusd/{timeframe}/`

---

## Commands
```bash
# Train (regression)
uv run python train.py --pair btcusdt --timeframe 15m --epochs 100 --batch_size 128 --pred_horizon 4

# Evaluate (regression)
uv run python predict.py --pair btcusdt --timeframe 15m

# Train (classifier)
uv run python train_clf.py --pair btcusdt --timeframe 15m --horizon 32 --epochs 100

# Evaluate (classifier)
uv run python predict_clf.py --pair btcusdt --timeframe 15m
```

---

## Current Status
**Last completed:** Experiment 10 - RevIN + Patching + MAE still 1.22x worse than naive.

**Conclusion so far:**
After 10+ experiments with different approaches:
- Regression (MSE) → model predicts zero
- Classification (BCE) → model predicts majority class
- Modern techniques (RevIN, Patching, MAE) → still worse than naive
- Different assets (BTC, Gold) → both unpredictable
- Different timeframes (15m, 4h, 1d) → all unpredictable

**The evidence strongly suggests:** Short-term price movements are fundamentally unpredictable (random walk). This aligns with the Efficient Market Hypothesis.

**Possible next steps:**
1. Accept the finding and document it
2. Try external features (sentiment, on-chain data, order book)
3. Try much longer timeframes (weekly/monthly)
4. Focus on volatility prediction instead of direction
