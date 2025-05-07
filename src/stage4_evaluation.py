"""
Stage 4: Evaluation
This module handles the evaluation of prediction models using various metrics.
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    """
    Class for evaluating prediction models using various metrics
    """
    
    def __init__(self, config):
        """
        Initialize the ModelEvaluator with configuration parameters
        
        Args:
            config: Configuration object containing evaluation parameters
        """
        self.config = config
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistical evaluation metrics
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            
        Returns:
            Dictionary containing various evaluation metrics
        """
        # สถิติพื้นฐาน
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Directional Accuracy
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
        
        # Print results
        print(f"\nStatistical Metrics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²: {r2:.6f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                initial_balance: float = None) -> Dict[str, Any]:
        """
        Calculate metrics related to trading performance
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            initial_balance: Initial trading balance
            
        Returns:
            Dictionary containing trading performance metrics
        """
        if initial_balance is None:
            initial_balance = self.config.INITIAL_BALANCE
            
        # คำนวณทิศทางการเปลี่ยนแปลงของราคา
        price_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)
        
        # สร้างสัญญาณการซื้อขาย (1 สำหรับซื้อ, -1 สำหรับขาย)
        signals = np.sign(pred_changes)
        
        # คำนวณกำไร/ขาดทุนจากการเทรด
        returns = signals[:-1] * price_changes[1:]
        cumulative_returns = np.cumsum(returns)
        
        # คำนวณอัตราผลตอบแทนประจำปี (Annualized Return)
        trading_days = len(returns)
        annual_return = (np.sum(returns) / initial_balance) * (252 / trading_days) * 100
        
        # Win Rate
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Sharpe Ratio (using daily returns)
        daily_returns = returns / initial_balance
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        
        # Maximum Drawdown
        balance_curve = initial_balance + cumulative_returns
        peaks = np.maximum.accumulate(balance_curve)
        drawdowns = (peaks - balance_curve) / peaks * 100
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Average Profit per Trade
        avg_profit = np.mean(returns) if len(returns) > 0 else 0
        avg_winning_trade = np.mean(returns[returns > 0]) if np.sum(returns > 0) > 0 else 0
        avg_losing_trade = np.mean(returns[returns < 0]) if np.sum(returns < 0) > 0 else 0
        
        # Risk-Reward Ratio
        risk_reward = abs(avg_winning_trade / avg_losing_trade) if avg_losing_trade != 0 else np.inf
        
        # Print results
        print(f"\nTrading Metrics:")
        print(f"  Annual Return: {annual_return:.2f}%")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Total Trades: {total_trades}")
        print(f"  Average Profit: {avg_profit:.6f}")
        print(f"  Risk-Reward Ratio: {risk_reward:.2f}")
        print(f"  Final Balance: {balance_curve[-1]:.2f} (from {initial_balance})")
        
        return {
            'annual_return': annual_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'avg_profit': avg_profit,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade, 
            'risk_reward_ratio': risk_reward,
            'cumulative_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0,
            'balance_curve': balance_curve,
            'drawdown_curve': drawdowns,
            'returns': returns,
            'signals': signals
        }
    
    def compare_with_benchmarks(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compare model performance with benchmark strategies
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            
        Returns:
            Dictionary containing benchmark comparison metrics
        """
        # กลยุทธ์ buy-and-hold
        buy_hold_return = (y_true[-1] - y_true[0]) / y_true[0] * 100
        
        # กลยุทธ์ SMA Crossover (5 และ 20 วัน)
        sma5 = np.array([np.mean(y_true[max(0, i-5):i]) if i > 0 else y_true[0] for i in range(len(y_true))])
        sma20 = np.array([np.mean(y_true[max(0, i-20):i]) if i > 0 else y_true[0] for i in range(len(y_true))])
        
        # สร้างสัญญาณ (1 เมื่อ sma5 > sma20, -1 เมื่อ sma5 < sma20)
        sma_signals = np.sign(sma5 - sma20)
        
        # คำนวณผลตอบแทน
        price_changes = np.diff(y_true)
        sma_returns = np.sum(sma_signals[:-1] * price_changes)
        sma_return = sma_returns / y_true[0] * 100
        
        # กลยุทธ์สุ่ม
        np.random.seed(42)  # กำหนด seed เพื่อให้ผลลัพธ์เหมือนเดิม
        random_signals = np.random.choice([-1, 1], size=len(y_true)-1)
        random_returns = np.sum(random_signals * price_changes)
        random_return = random_returns / y_true[0] * 100
        
        # กลยุทธ์ของโมเดล
        model_signals = np.sign(np.diff(y_pred))
        model_returns = np.sum(model_signals[:-1] * price_changes[1:])
        model_return = model_returns / y_true[0] * 100
        
        # Print results
        print(f"\nBenchmark Comparison:")
        print(f"  Model Return: {model_return:.2f}%")
        print(f"  Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"  SMA Crossover Return: {sma_return:.2f}%")
        print(f"  Random Strategy Return: {random_return:.2f}%")
        
        # Calculate relative performance
        model_vs_buyhold = (model_return - buy_hold_return) / abs(buy_hold_return) * 100 if buy_hold_return != 0 else np.inf
        model_vs_sma = (model_return - sma_return) / abs(sma_return) * 100 if sma_return != 0 else np.inf
        
        print(f"  Model outperformed Buy & Hold by: {model_vs_buyhold:.2f}%")
        print(f"  Model outperformed SMA Crossover by: {model_vs_sma:.2f}%")
        
        return {
            'buy_hold_return': buy_hold_return,
            'sma_return': sma_return,
            'random_return': random_return,
            'model_return': model_return,
            'model_vs_buyhold': model_vs_buyhold,
            'model_vs_sma': model_vs_sma
        }
    
    def forward_walking_validation(model, X, y, window_size=500, step=100):
        predictions = []
        actuals = []
        
        for i in range(0, len(X) - window_size, step):
            # ฝึกฝนบนหน้าต่าง
            X_train = X[i:i+window_size]
            y_train = y[i:i+window_size]
            
            # ทำนายจุดถัดไป
            X_test = X[i+window_size:i+window_size+step]
            y_test = y[i+window_size:i+window_size+step]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            predictions.extend(preds)
            actuals.extend(y_test)
        
        return np.array(predictions), np.array(actuals)
    
    def directional_accuracy_at_key_levels(y_true, y_pred, prices, threshold=0.01):
        # ระบุระดับสำคัญ (วิธีอย่างง่าย)
        highs = prices.rolling(20).max()
        lows = prices.rolling(20).min()
        
        # ตรวจสอบว่าราคาอยู่ใกล้ระดับสำคัญหรือไม่ (ภายในเกณฑ์)
        near_key_level = (
            (np.abs(prices - highs) / prices < threshold) | 
            (np.abs(prices - lows) / prices < threshold)
        )
        
        # คำนวณความแม่นยำในการทำนายทิศทางเฉพาะที่จุดเหล่านี้
        if sum(near_key_level) > 0:
            direction_true = np.sign(np.diff(y_true))
            direction_pred = np.sign(np.diff(y_pred))
            
            # กรองสำหรับระดับสำคัญ
            key_level_idx = near_key_level[:-1]  # ปรับสำหรับความยาว diff()
            key_level_accuracy = np.mean(
                direction_true[key_level_idx] == direction_pred[key_level_idx]
            ) * 100
            
            return key_level_accuracy
        return None