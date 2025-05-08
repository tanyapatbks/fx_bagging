# Summary Report - EURUSD (E)

## 1. RMSE Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 0.627043 | 0.625991 | 0.627172 | 
| GRU | 0.626430 | 0.626654 | 0.628063 | 
| XGBoost | 0.491987 | 0.492188 | 0.492185 | 
| TFT | 0.626916 | 0.620137 | 0.624889 | 

## 2. Directional Accuracy Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 48.00% | 48.48% | 47.71% | 
| GRU | 48.64% | 46.65% | 47.00% | 
| XGBoost | 49.66% | 49.15% | 48.77% | 
| TFT | 48.38% | 46.97% | 46.46% | 

## 3. Annual Return Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | -0.00% | -0.00% | -0.00% | 
| GRU | -0.00% | -0.00% | -0.00% | 
| XGBoost | -0.00% | 0.00% | 0.00% | 
| TFT | -0.00% | -0.00% | -0.00% | 

## 4. Model Return vs Benchmark Comparison (Enhanced+Selected Data)

| Model | Model Return | Buy & Hold | SMA Crossover | Random |
|-------|-------------|-----------|---------------|--------|
| LSTM | -121.53% | -79.38% | -27.33% | 80.85% |
| GRU | -84.22% | -79.38% | -27.33% | 80.85% |
| XGBoost | 21.08% | -79.38% | -27.33% | 80.85% |
| TFT | -39.63% | -79.38% | -27.33% | 80.85% |

## 5. Key Findings

- Best model for RMSE: **XGBoost** (0.492185)
- Best model for Directional Accuracy: **XGBoost** (48.77%)
- Best model for Annual Return: **XGBoost** (0.00%)

### Improvement from Raw to Enhanced+Selected

- LSTM: RMSE improvement: **-0.02%**
- GRU: RMSE improvement: **-0.26%**
- XGBoost: RMSE improvement: **-0.04%**
- TFT: RMSE improvement: **0.32%**
