# Summary Report - GBPUSD (G)

## 1. RMSE Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 0.736310 | 0.733823 | 0.735017 | 
| GRU | 0.736385 | 0.737315 | 0.737599 | 
| XGBoost | 0.594650 | 0.594557 | 0.594659 | 
| TFT | 0.736688 | 0.732942 | 0.731956 | 

## 2. Directional Accuracy Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 48.38% | 49.28% | 48.80% | 
| GRU | 49.18% | 48.32% | 48.48% | 
| XGBoost | 50.08% | 50.56% | 50.66% | 
| TFT | 48.96% | 48.09% | 48.06% | 

## 3. Annual Return Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | -0.00% | 0.00% | 0.00% | 
| GRU | 0.00% | 0.00% | 0.00% | 
| XGBoost | -0.00% | 0.00% | 0.00% | 
| TFT | 0.00% | -0.00% | 0.00% | 

## 4. Model Return vs Benchmark Comparison (Enhanced+Selected Data)

| Model | Model Return | Buy & Hold | SMA Crossover | Random |
|-------|-------------|-----------|---------------|--------|
| LSTM | 151.42% | -45.23% | 67.69% | 7.76% |
| GRU | 17.47% | -45.23% | 67.69% | 7.76% |
| XGBoost | 16.86% | -45.23% | 67.69% | 7.76% |
| TFT | 64.14% | -45.23% | 67.69% | 7.76% |

## 5. Key Findings

- Best model for RMSE: **XGBoost** (0.594659)
- Best model for Directional Accuracy: **XGBoost** (50.66%)
- Best model for Annual Return: **LSTM** (0.00%)

### Improvement from Raw to Enhanced+Selected

- LSTM: RMSE improvement: **0.18%**
- GRU: RMSE improvement: **-0.16%**
- XGBoost: RMSE improvement: **-0.00%**
- TFT: RMSE improvement: **0.64%**
