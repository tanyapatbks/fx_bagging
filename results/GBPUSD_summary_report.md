# Summary Report - GBPUSD (G)

## 1. RMSE Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 0.737240 | 0.734419 | 0.732910 | 
| GRU | 0.736272 | 0.735647 | 0.734826 | 
| XGBoost | 0.594302 | 0.594306 | 0.594287 | 
| TFT | 0.735888 | 0.731716 | 0.733569 | 

## 2. Directional Accuracy Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 49.76% | 49.50% | 49.66% | 
| GRU | 50.14% | 48.16% | 48.57% | 
| XGBoost | 50.18% | 50.24% | 49.79% | 
| TFT | 48.80% | 47.87% | 48.28% | 

## 3. Annual Return Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | -0.00% | 0.00% | 0.00% | 
| GRU | 0.00% | 0.00% | 0.00% | 
| XGBoost | 0.00% | 0.00% | 0.00% | 
| TFT | 0.00% | -0.00% | 0.00% | 

## 4. Model Return vs Benchmark Comparison (Enhanced+Selected Data)

| Model | Model Return | Buy & Hold | SMA Crossover | Random |
|-------|-------------|-----------|---------------|--------|
| LSTM | 91.58% | -45.23% | 67.69% | 7.76% |
| GRU | 62.68% | -45.23% | 67.69% | 7.76% |
| XGBoost | 46.69% | -45.23% | 67.69% | 7.76% |
| TFT | 39.03% | -45.23% | 67.69% | 7.76% |

## 5. Key Findings

- Best model for RMSE: **XGBoost** (0.594287)
- Best model for Directional Accuracy: **XGBoost** (49.79%)
- Best model for Annual Return: **LSTM** (0.00%)

### Improvement from Raw to Enhanced+Selected

- LSTM: RMSE improvement: **0.59%**
- GRU: RMSE improvement: **0.20%**
- XGBoost: RMSE improvement: **0.00%**
- TFT: RMSE improvement: **0.32%**
