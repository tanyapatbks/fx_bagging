# Summary Report - USDJPY (J)

## 1. RMSE Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 106.342496 | 106.949676 | 106.507216 | 
| GRU | 106.288355 | 106.586313 | 106.523570 | 
| XGBoost | 83.008375 | 83.007012 | 83.007348 | 
| TFT | 106.211539 | 106.386611 | 106.914092 | 

## 2. Directional Accuracy Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 49.82% | 51.33% | 50.69% | 
| GRU | 49.95% | 48.48% | 48.67% | 
| XGBoost | 48.16% | 48.12% | 48.48% | 
| TFT | 49.82% | 48.00% | 48.16% | 

## 3. Annual Return Comparison

| Model | Data Type 1 (Raw) | Data Type 2 (Enhanced) | Data Type 3 (Enhanced+Selected) |
|-------|-------------------|------------------------|--------------------------------|
| LSTM | 0.00% | 0.00% | 0.00% | 
| GRU | -0.00% | 0.00% | 0.00% | 
| XGBoost | -0.00% | 0.00% | 0.00% | 
| TFT | 0.00% | 0.00% | 0.00% | 

## 4. Model Return vs Benchmark Comparison (Enhanced+Selected Data)

| Model | Model Return | Buy & Hold | SMA Crossover | Random |
|-------|-------------|-----------|---------------|--------|
| LSTM | 164.00% | 118.52% | 266.99% | -250.76% |
| GRU | 174.01% | 118.52% | 266.99% | -250.76% |
| XGBoost | 3.68% | 118.52% | 266.99% | -250.76% |
| TFT | 94.16% | 118.52% | 266.99% | -250.76% |

## 5. Key Findings

- Best model for RMSE: **XGBoost** (83.007348)
- Best model for Directional Accuracy: **LSTM** (50.69%)
- Best model for Annual Return: **GRU** (0.00%)

### Improvement from Raw to Enhanced+Selected

- LSTM: RMSE improvement: **-0.15%**
- GRU: RMSE improvement: **-0.22%**
- XGBoost: RMSE improvement: **0.00%**
- TFT: RMSE improvement: **-0.66%**
