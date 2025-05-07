# Summary Report - USDJPY (J)

## 1. RMSE Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 107.838181 | 108.270658 | 107.799817 | 
| GRU | 107.800876 | 107.832911 | 107.774068 | 
| XGBoost | 106.837182 | 106.817200 | 106.819116 | 
| TFT | 107.781900 | 108.234197 | 108.260171 | 

## 2. Directional Accuracy Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 45.63% | 48.21% | 49.40% | 
| GRU | 48.61% | 47.62% | 48.41% | 
| XGBoost | 50.99% | 51.19% | 51.39% | 
| TFT | 49.80% | 49.80% | 50.99% | 

## 3. Annual Return Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | -0.00% | 0.00% | 0.00% | 
| GRU | -0.00% | 0.00% | 0.00% | 
| XGBoost | 0.00% | -0.00% | 0.00% | 
| TFT | 0.00% | -0.00% | 0.00% | 

## 4. Model Return vs Benchmark Comparison (Enhanced+Selected Data)

| Model | Model Return | Buy & Hold | SMA Crossover | Random |
|-------|-------------|-----------|---------------|--------|
| LSTM | 12.38% | -100.00% | -14.40% | 6.40% |
| GRU | 104.48% | -100.00% | -14.40% | 6.40% |
| XGBoost | 12.21% | -100.00% | -14.40% | 6.40% |
| TFT | 64.25% | -100.00% | -14.40% | 6.40% |

## 5. Key Findings

- Best model for RMSE: **XGBoost** (106.819116)
- Best model for Directional Accuracy: **XGBoost** (51.39%)
- Best model for Annual Return: **GRU** (0.00%)

### Improvement from Raw to Enhanced+Selected

- LSTM: RMSE improvement: **0.04%**
- GRU: RMSE improvement: **0.02%**
- XGBoost: RMSE improvement: **0.02%**
- TFT: RMSE improvement: **-0.44%**
