# Summary Report - GBPUSD (G)

## 1. RMSE Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 0.831990 | 0.830040 | 0.828019 | 
| GRU | 0.832410 | 0.828234 | 0.830542 | 
| XGBoost | 0.872607 | 0.872593 | 0.872570 | 
| TFT | 0.831472 | 0.828723 | 0.830601 | 

## 2. Directional Accuracy Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 51.19% | 48.81% | 49.01% | 
| GRU | 50.20% | 51.79% | 53.57% | 
| XGBoost | 46.03% | 47.02% | 46.43% | 
| TFT | 50.79% | 50.20% | 51.19% | 

## 3. Annual Return Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | -0.00% | -0.00% | -0.00% | 
| GRU | -0.00% | -0.00% | -0.00% | 
| XGBoost | 0.00% | 0.00% | -0.00% | 
| TFT | -0.00% | -0.00% | 0.00% | 

## 4. Model Return vs Benchmark Comparison (Enhanced+Selected Data)

| Model | Model Return | Buy & Hold | SMA Crossover | Random |
|-------|-------------|-----------|---------------|--------|
| LSTM | -260.27% | 588.22% | -319.63% | -79.36% |
| GRU | -109.95% | 588.22% | -319.63% | -79.36% |
| XGBoost | -105.30% | 588.22% | -319.63% | -79.36% |
| TFT | 10.59% | 588.22% | -319.63% | -79.36% |

## 5. Key Findings

- Best model for RMSE: **LSTM** (0.828019)
- Best model for Directional Accuracy: **GRU** (53.57%)
- Best model for Annual Return: **TFT** (0.00%)

### Improvement from Raw to Enhanced+Selected

- LSTM: RMSE improvement: **0.48%**
- GRU: RMSE improvement: **0.22%**
- XGBoost: RMSE improvement: **0.00%**
- TFT: RMSE improvement: **0.10%**
