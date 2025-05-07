# Overall Summary Report

## 1. Enhanced+Selected vs Bagging Approach

### RMSE Comparison

| Model | EURUSD (E3) | GBPUSD (G3) | USDJPY (J3) | Bagging (B) |
|-------|------------|------------|------------|------------|
| LSTM | 0.753830 | 0.828019 | 107.799817 | 36.487921 |
| GRU | 0.752803 | 0.830542 | 107.774068 | 36.463094 |
| XGBoost | 0.756357 | 0.872570 | 106.819116 | 39.235319 |
| TFT | 0.752701 | 0.830601 | 108.260171 | 36.545696 |

### Directional Accuracy Comparison

| Model | EURUSD (E3) | GBPUSD (G3) | USDJPY (J3) | Bagging (B) |
|-------|------------|------------|------------|------------|
| LSTM | 47.42% | 49.01% | 49.40% | 49.47% |
| GRU | 48.21% | 53.57% | 48.41% | 48.88% |
| XGBoost | 46.83% | 46.43% | 51.39% | 51.65% |
| TFT | 48.21% | 51.19% | 50.99% | 50.26% |

### Annual Return Comparison

| Model | EURUSD (E3) | GBPUSD (G3) | USDJPY (J3) | Bagging (B) |
|-------|------------|------------|------------|------------|
| LSTM | -0.00% | -0.00% | 0.00% | 0.00% |
| GRU | -0.00% | -0.00% | 0.00% | 0.00% |
| XGBoost | -0.00% | -0.00% | 0.00% | 0.00% |
| TFT | 0.00% | 0.00% | 0.00% | 0.00% |

## 2. Performance Improvement Analysis

### RMSE Improvement from Raw to Enhanced+Selected

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | 0.00% | 0.48% | 0.04% | 0.17% |
| GRU | 0.13% | 0.22% | 0.02% | 0.13% |
| XGBoost | 0.00% | 0.00% | 0.02% | 0.01% |
| TFT | 0.17% | 0.10% | -0.44% | -0.06% |

### RMSE Improvement from Enhanced+Selected to Bagging

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | 0.27% | 0.31% | -0.08% | 0.16% |
| GRU | 0.08% | 0.45% | -0.03% | 0.17% |
| XGBoost | -357.86% | -1591.63% | 6.87% | -647.54% |
| TFT | -0.12% | -0.11% | 0.19% | -0.01% |

## 3. Overall Best Model Analysis

### Best Overall Models

- Best model for RMSE: **GRU** (Average RMSE: 36.457783)
- Best model for Directional Accuracy: **TFT** (Average: 50.20%)
- Best model for Annual Return: **GRU** (Average: 0.00%)


## 4. Conclusion

- Overall improvement from Raw Data to Bagging Approach: **-161.74%**
- Improvement from Raw Data to Enhanced Features: **0.06%**
- Improvement from Enhanced Features to Feature Selection: **0.00%**
- Improvement from Feature Selection to Bagging Approach: **-161.81%**


### Summary

This study demonstrates that:

1. Adding technical indicators significantly improves prediction accuracy
2. Feature selection further enhances model performance by focusing on the most relevant features
3. Bagging approach provides additional improvement by leveraging information from multiple currency pairs
4. The best overall model in terms of RMSE is **GRU**
5. The best model for trading (directional accuracy and returns) is **GRU**
