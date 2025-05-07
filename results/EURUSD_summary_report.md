# Summary Report - EURUSD (E)

## 1. RMSE Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 0.627559 | 0.626504 | 0.625684 | 
| GRU | 0.626802 | 0.628078 | 0.627286 | 
| XGBoost | 0.493064 | 0.493251 | 0.493225 | 
| TFT | 0.627231 | 0.628288 | 0.626716 | 

## 2. Directional Accuracy Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 48.73% | 47.32% | 47.96% | 
| GRU | 48.06% | 47.10% | 46.39% | 
| XGBoost | 48.99% | 48.93% | 49.34% | 
| TFT | 48.93% | 46.55% | 47.19% | 

## 3. Annual Return Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | -0.00% | -0.00% | -0.00% | 
| GRU | -0.00% | -0.00% | -0.00% | 
| XGBoost | 0.00% | 0.00% | 0.00% | 
| TFT | -0.00% | -0.00% | -0.00% | 

## 4. Model Return vs Benchmark Comparison (Enhanced+Selected Data)

| Model | Model Return | Buy & Hold | SMA Crossover | Random |
|-------|-------------|-----------|---------------|--------|
| LSTM | -43.04% | -79.38% | -27.33% | 80.85% |
| GRU | -84.22% | -79.38% | -27.33% | 80.85% |
| XGBoost | 14.45% | -79.38% | -27.33% | 80.85% |
| TFT | -84.87% | -79.38% | -27.33% | 80.85% |

## 5. Key Findings

- Best model for RMSE: **XGBoost** (0.493225)
- Best model for Directional Accuracy: **XGBoost** (49.34%)
- Best model for Annual Return: **XGBoost** (0.00%)

### Improvement from Raw to Enhanced+Selected

- LSTM: RMSE improvement: **0.30%**
- GRU: RMSE improvement: **-0.08%**
- XGBoost: RMSE improvement: **-0.03%**
- TFT: RMSE improvement: **0.08%**
