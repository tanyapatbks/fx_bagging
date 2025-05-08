# Bagging Approach Summary Report

## 1. RMSE Comparison

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | 0.619200 | 0.731873 | 106.016665 | 35.789246 |
| GRU | 0.626897 | 0.735035 | 106.170889 | 35.844274 |
| XGBoost | 3.980575 | 3.811968 | 58.575080 | 22.122541 |
| TFT | 0.618500 | 0.732006 | 106.069767 | 35.806758 |

## 2. Directional Accuracy Comparison

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | 48.48% | 49.54% | 50.91% | 49.64% |
| GRU | 46.71% | 48.99% | 49.15% | 48.28% |
| XGBoost | 49.66% | 49.18% | 51.49% | 50.11% |
| TFT | 47.03% | 48.41% | 47.87% | 47.77% |

## 3. Annual Return Comparison

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | -0.00% | 0.00% | 0.00% | 0.00% |
| GRU | -0.00% | 0.00% | 0.00% | 0.00% |
| XGBoost | -0.00% | -0.00% | -0.00% | -0.00% |
| TFT | -0.00% | 0.00% | 0.00% | 0.00% |

## 4. Key Findings

- Best model for RMSE: **XGBoost** (22.122541)
- Best model for Directional Accuracy: **XGBoost** (50.11%)
- Best model for Annual Return: **GRU** (0.00%)

### Bagging Approach Benefits

- Bagging combines predictions from models trained on different currency pairs
- This approach helps to capture universal forex market patterns
- Reduces overfitting to patterns specific to a single currency pair
- Can improve robustness and generalization of predictions
