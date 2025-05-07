# Bagging Approach Summary Report

## 1. RMSE Comparison

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | 0.629944 | 0.735059 | 106.157750 | 35.840918 |
| GRU | 0.627868 | 0.734504 | 106.198832 | 35.853735 |
| XGBoost | 3.630049 | 3.385251 | 61.403100 | 22.806133 |
| TFT | 0.619692 | 0.733006 | 106.068558 | 35.807085 |

## 2. Directional Accuracy Comparison

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | 47.45% | 48.73% | 49.66% | 48.62% |
| GRU | 47.74% | 48.19% | 49.50% | 48.48% |
| XGBoost | 52.07% | 49.47% | 51.72% | 51.08% |
| TFT | 46.59% | 47.23% | 46.33% | 46.71% |

## 3. Annual Return Comparison

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | -0.00% | 0.00% | 0.00% | -0.00% |
| GRU | 0.00% | 0.00% | 0.00% | 0.00% |
| XGBoost | -0.00% | 0.00% | -0.00% | 0.00% |
| TFT | -0.00% | 0.00% | 0.00% | 0.00% |

## 4. Key Findings

- Best model for RMSE: **XGBoost** (22.806133)
- Best model for Directional Accuracy: **XGBoost** (51.08%)
- Best model for Annual Return: **GRU** (0.00%)

### Bagging Approach Benefits

- Bagging combines predictions from models trained on different currency pairs
- This approach helps to capture universal forex market patterns
- Reduces overfitting to patterns specific to a single currency pair
- Can improve robustness and generalization of predictions
