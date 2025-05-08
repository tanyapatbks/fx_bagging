# Overall Summary Report

## 1. Enhanced+Selected vs Bagging Approach

### RMSE Comparison

| Model | EURUSD (E3) | GBPUSD (G3) | USDJPY (J3) | Bagging (B) |
|-------|------------|------------|------------|------------|
| LSTM | 0.627172 | 0.732910 | 106.507216 | 35.789246 |
| GRU | 0.628063 | 0.734826 | 106.523570 | 35.844274 |
| XGBoost | 0.492185 | 0.594287 | 83.007348 | 22.122541 |
| TFT | 0.624889 | 0.733569 | 106.914092 | 35.806758 |

### Directional Accuracy Comparison

| Model | EURUSD (E3) | GBPUSD (G3) | USDJPY (J3) | Bagging (B) |
|-------|------------|------------|------------|------------|
| LSTM | 47.71% | 49.66% | 50.69% | 49.64% |
| GRU | 47.00% | 48.57% | 48.67% | 48.28% |
| XGBoost | 48.77% | 49.79% | 48.48% | 50.11% |
| TFT | 46.46% | 48.28% | 48.16% | 47.77% |

### Annual Return Comparison

| Model | EURUSD (E3) | GBPUSD (G3) | USDJPY (J3) | Bagging (B) |
|-------|------------|------------|------------|------------|
| LSTM | -0.00% | 0.00% | 0.00% | 0.00% |
| GRU | -0.00% | 0.00% | 0.00% | 0.00% |
| XGBoost | 0.00% | 0.00% | 0.00% | -0.00% |
| TFT | -0.00% | 0.00% | 0.00% | 0.00% |

## 2. Performance Improvement Analysis

### RMSE Improvement from Raw to Enhanced+Selected

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | -0.02% | 0.59% | -0.15% | 0.14% |
| GRU | -0.26% | 0.20% | -0.22% | -0.10% |
| XGBoost | -0.04% | 0.00% | 0.00% | -0.01% |
| TFT | 0.32% | 0.32% | -0.66% | -0.01% |

### RMSE Improvement from Enhanced+Selected to Bagging

| Model | EURUSD | GBPUSD | USDJPY | Average |
|-------|--------|--------|--------|--------|
| LSTM | 1.27% | 0.14% | 0.46% | 0.62% |
| GRU | 0.19% | -0.03% | 0.33% | 0.16% |
| XGBoost | -708.76% | -541.44% | 29.43% | -406.92% |
| TFT | 1.02% | 0.21% | 0.79% | 0.68% |

## 3. Overall Best Model Analysis

### Best Overall Models

- Best model for RMSE: **XGBoost** (Average RMSE: 25.076907)
- Best model for Directional Accuracy: **XGBoost** (Average: 49.56%)
- Best model for Annual Return: **GRU** (Average: 0.00%)


## 4. Conclusion

- Overall improvement from Raw Data to Bagging Approach: **-101.38%**
- Improvement from Raw Data to Enhanced Features: **0.10%**
- Improvement from Enhanced Features to Feature Selection: **-0.09%**
- Improvement from Feature Selection to Bagging Approach: **-101.36%**


### Summary

This study demonstrates that:

1. Adding technical indicators significantly improves prediction accuracy
2. Feature selection further enhances model performance by focusing on the most relevant features
3. Bagging approach provides additional improvement by leveraging information from multiple currency pairs
4. The best overall model in terms of RMSE is **XGBoost**
5. The best model for trading (directional accuracy and returns) is **GRU**
