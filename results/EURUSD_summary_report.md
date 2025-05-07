# Summary Report - EURUSD (E)

## 1. RMSE Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 0.753867 | 0.752848 | 0.753830 | 
| GRU | 0.753775 | 0.751887 | 0.752803 | 
| XGBoost | 0.756358 | 0.756360 | 0.756357 | 
| TFT | 0.753999 | 0.753078 | 0.752701 | 

## 2. Directional Accuracy Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 47.82% | 48.41% | 47.42% | 
| GRU | 47.62% | 48.21% | 48.21% | 
| XGBoost | 46.63% | 46.43% | 46.83% | 
| TFT | 45.83% | 49.40% | 48.21% | 

## 3. Annual Return Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | -0.00% | -0.00% | -0.00% | 
| GRU | -0.00% | -0.00% | -0.00% | 
| XGBoost | -0.00% | -0.00% | -0.00% | 
| TFT | -0.00% | -0.00% | 0.00% | 

## 4. Model Return vs Benchmark Comparison (Enhanced+Selected Data)

| Model | Model Return | Buy & Hold | SMA Crossover | Random |
|-------|-------------|-----------|---------------|--------|
| LSTM | -819.38% | 1279.79% | 214.38% | 74.38% |
| GRU | -425.21% | 1279.79% | 214.38% | 74.38% |
| XGBoost | -15.00% | 1279.79% | 214.38% | 74.38% |
| TFT | 35.63% | 1279.79% | 214.38% | 74.38% |

## 5. Key Findings

- Best model for RMSE: **TFT** (0.752701)
- Best model for Directional Accuracy: **GRU** (48.21%)
- Best model for Annual Return: **TFT** (0.00%)

### Improvement from Raw to Enhanced+Selected

- LSTM: RMSE improvement: **0.00%**
- GRU: RMSE improvement: **0.13%**
- XGBoost: RMSE improvement: **0.00%**
- TFT: RMSE improvement: **0.17%**
