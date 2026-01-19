# NVIDIA (NVDA) Stock Price Prediction with BiLSTM (Deep Learning)

A portfolio-style time-series project that trains a **Bidirectional LSTM (BiLSTM)** model to predict **NVDA’s next-day closing price** from historical daily market data and engineered technical features.

> **Disclaimer:** This project is for educational purposes only and is **not** financial advice.

---

## Overview
- Pulling daily NVDA data (**tries Alpaca first**, falls back to **Yahoo Finance via `yfinance`**)
- Feature engineering (moving averages + volatility + volume)
- Building 30-day sequences for an LSTM model
- Training and evaluating a **BiLSTM** with a **chronological (time-based)** split
- Producing a **Predicted vs True** test plot
- Generating a **next-day price prediction** using the most recent window

---

## Dataset
- **Symbol:** NVDA  
- **Frequency:** Daily bars  
- **Rows fetched:** 1014  
- **Rows after feature engineering:** 994  

If Alpaca is not available, the notebook automatically falls back to `yfinance`.

---

## Features & Target

### Input features (6)
| Feature | Description |
|---|---|
| `close` | Daily closing price |
| `volume` | Daily traded volume |
| `volatility` | Rolling std. dev of daily returns (5-day window) |
| `ma_5` | 5-day moving average of close |
| `ma_10` | 10-day moving average of close |
| `ma_20` | 20-day moving average of close |

### Target
- **Next-day closing price**: `target = close(t+1)`

---

## Time-Series Setup (No Leakage)

### Sliding window sequences
- **Lookback window:** 30 trading days  
- Each sample = last 30 days of features → predicts next-day close

### Train/Test split (chronological)
The split is done by **time order**, not random shuffling:

- **Train:** first ~80% of sequences  
- **Test:** last ~20% of sequences (unseen future period)

Shapes from the run:
- Train: `(772, 30, 6)`
- Test: `(192, 30, 6)`

---

## Model Architecture
**Bidirectional LSTM → Dropout → Bidirectional LSTM → Dropout → Dense(16) → Dense(1)**

Total parameters: **78,625**

Loss: **MSE**  
Metric: **MAE**

---

## Results (latest run)
- **Test MAE:** `6.9718 USD`
- **Last close (most recent):** `187.05`
- **Model next-day predicted close:** `180.95`

Interpretation: The model captures the **overall trend** but produces **smoother predictions**, under-reacting to sharp spikes/drops (common for price regression using MSE).

---

## Plot: Predictions vs True (Test Set)

This plot compares model predictions vs actual close prices over the unseen test window (~9 months).

![LSTM Predictions vs True (Test Set)](images/lstm_predictions_vs_true.png)

---

## How to Run

### Option A — Run in Jupyter / Colab (recommended)
1. Open `Nvidia_stock_prediction.ipynb`
2. Run all cells from top to bottom.

### Option B — Run locally
#### 1) Install dependencies
```bash
pip install -r requirements.txt
