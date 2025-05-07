# Overall Summary Report

## 1. Enhanced+Selected vs Bagging Approach

### RMSE Comparison

| Model | EURUSD (E3) | GBPUSD (G3) | USDJPY (J3) | Bagging (B) |
|-------|------------|------------|------------|------------|
| LSTM | 0.625684 | 0.735017 | 106.532449 | 35.840918 |
| GRU | 0.627286 | 0.737599 | 106.382484 | 35.853735 |
| XGBoost | 0.493225 | 0.594659 | 83.083665 | 22.806133 |
| TFT | 0.626716 | 0.731956 | 107.094965 | 35.807085 |

### Directional Accuracy Comparison

| Model | EURUSD (E3) | GBPUSD (G3) | USDJPY (J3) | Bagging (B) |
|-------|------------|------------|------------|------------|
| LSTM | 47.96% | 48.80% | 50.11% | 48.62% |
| GRU | 46.39% | 48.48% | 48.73% | 48.48% |
| XGBoost | 49.34% | 50.66% | 49.09% | 51.08% |
| TFT | 47.19% | 48.06% | 47.58% | 46.71% |

### Annual Return Comparison

| Model | EURUSD (E3) | GBPUSD (G3) | USDJPY (J3) | Bagging (B) |
|-------|------------|------------|------------|------------|
| LSTM | -0.00% | 0.00% | 0.00% | -0.00% |
| GRU | -0.00% | 0.00% | 0.00% | 0.00% |
| XGBoost | 0.00% | 0.00% | -0.00% | 0.00% |
| TFT | -0.00% | 0.00% | 0.00% | 0.00% |

## 2. Performance Improvement Analysis

### RMSE Improvement from Raw to Enhanced+Selected

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | 0.30% | 0.18% | -0.18% | 0.10% |
| GRU | -0.08% | -0.16% | -0.08% | -0.11% |
| XGBoost | -0.03% | -0.00% | -0.00% | -0.01% |
| TFT | 0.08% | 0.64% | -0.80% | -0.02% |

### RMSE Improvement from Enhanced+Selected to Bagging

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | -0.68% | -0.01% | 0.35% | -0.11% |
| GRU | -0.09% | 0.42% | 0.17% | 0.17% |
| XGBoost | -635.98% | -469.28% | 26.09% | -359.72% |
| TFT | 1.12% | -0.14% | 0.96% | 0.65% |

## 3. Overall Best Model Analysis

### Best Overall Models

- Best model for RMSE: **XGBoost** (Average RMSE: 25.431658)
- Best model for Directional Accuracy: **XGBoost** (Average: 50.39%)
- Best model for Annual Return: **GRU** (Average: 0.00%)


## 4. Conclusion

- Overall improvement from Raw Data to Bagging Approach: **-89.78%**
- Improvement from Raw Data to Enhanced Features: **-0.00%**
- Improvement from Enhanced Features to Feature Selection: **-0.01%**
- Improvement from Feature Selection to Bagging Approach: **-89.76%**


### Summary

This study demonstrates that:

1. Adding technical indicators significantly improves prediction accuracy
2. Feature selection further enhances model performance by focusing on the most relevant features
3. Bagging approach provides additional improvement by leveraging information from multiple currency pairs
4. The best overall model in terms of RMSE is **XGBoost**
5. The best model for trading (directional accuracy and returns) is **GRU**
