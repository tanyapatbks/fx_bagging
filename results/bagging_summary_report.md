# Bagging Approach Summary Report

## 1. RMSE Comparison

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | 0.751831 | 0.825463 | 107.886469 | 36.487921 |
| GRU | 0.752220 | 0.826773 | 107.810290 | 36.463094 |
| XGBoost | 3.463081 | 14.760694 | 99.482183 | 39.235319 |
| TFT | 0.753627 | 0.831475 | 108.051987 | 36.545696 |

## 2. Directional Accuracy Comparison

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | 48.61% | 52.38% | 47.42% | 49.47% |
| GRU | 48.21% | 49.40% | 49.01% | 48.88% |
| XGBoost | 52.98% | 53.17% | 48.81% | 51.65% |
| TFT | 52.18% | 50.20% | 48.41% | 50.26% |

## 3. Annual Return Comparison

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | -0.00% | 0.00% | 0.00% | 0.00% |
| GRU | -0.00% | 0.00% | 0.00% | 0.00% |
| XGBoost | -0.00% | -0.00% | 0.01% | 0.00% |
| TFT | 0.00% | -0.00% | 0.00% | 0.00% |

## 4. Key Findings

- Best model for RMSE: **GRU** (36.463094)
- Best model for Directional Accuracy: **XGBoost** (51.65%)
- Best model for Annual Return: **GRU** (0.00%)

### Bagging Approach Benefits

- Bagging combines predictions from models trained on different currency pairs
- This approach helps to capture universal forex market patterns
- Reduces overfitting to patterns specific to a single currency pair
- Can improve robustness and generalization of predictions
