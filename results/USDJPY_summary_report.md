# Summary Report - USDJPY (J)

## 1. RMSE Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 106.338014 | 106.663784 | 106.532449 | 
| GRU | 106.297167 | 106.132023 | 106.382484 | 
| XGBoost | 83.081420 | 83.084364 | 83.083665 | 
| TFT | 106.249541 | 106.665183 | 107.094965 | 

## 2. Directional Accuracy Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 50.30% | 49.98% | 50.11% | 
| GRU | 50.30% | 49.18% | 48.73% | 
| XGBoost | 48.80% | 48.54% | 49.09% | 
| TFT | 48.67% | 47.26% | 47.58% | 

## 3. Annual Return Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 0.00% | 0.00% | 0.00% | 
| GRU | 0.00% | 0.00% | 0.00% | 
| XGBoost | 0.00% | 0.00% | -0.00% | 
| TFT | 0.00% | 0.00% | 0.00% | 

## 4. Model Return vs Benchmark Comparison (Enhanced+Selected Data)

| Model | Model Return | Buy & Hold | SMA Crossover | Random |
|-------|-------------|-----------|---------------|--------|
| LSTM | 115.98% | 118.52% | 266.99% | -250.76% |
| GRU | 183.74% | 118.52% | 266.99% | -250.76% |
| XGBoost | -32.38% | 118.52% | 266.99% | -250.76% |
| TFT | 90.97% | 118.52% | 266.99% | -250.76% |

## 5. Key Findings

- Best model for RMSE: **XGBoost** (83.083665)
- Best model for Directional Accuracy: **LSTM** (50.11%)
- Best model for Annual Return: **GRU** (0.00%)

### Improvement from Raw to Enhanced+Selected

- LSTM: RMSE improvement: **-0.18%**
- GRU: RMSE improvement: **-0.08%**
- XGBoost: RMSE improvement: **-0.00%**
- TFT: RMSE improvement: **-0.80%**
