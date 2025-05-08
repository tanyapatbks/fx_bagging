"""
Stage 4: Evaluation
This module handles the evaluation of prediction models using various metrics.
"""

import os
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


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
        
        # สร้างไดเรกทอรีสำหรับบันทึกผลการประเมิน
        self.results_path = os.path.join(config.RESULTS_PATH, "evaluation")
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         verbose: bool = True) -> Dict[str, float]:
        """
        Calculate statistical evaluation metrics
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            verbose: Whether to print results to console
            
        Returns:
            Dictionary containing various evaluation metrics
        """
        # ตรวจสอบข้อมูลนำเข้า
        if len(y_true) != len(y_pred):
            raise ValueError(f"Lengths of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) do not match")
        
        # ตรวจสอบและแก้ไข NaN, Inf
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # แจ้งเตือนถ้ามีการกรองข้อมูล
        if len(y_true_clean) < len(y_true):
            print(f"WARNING: Removed {len(y_true) - len(y_true_clean)} invalid values for metrics calculation")
        
        # สถิติพื้นฐาน
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        mask = y_true_clean != 0
        mape = np.mean(np.abs((y_true_clean[mask] - y_pred_clean[mask]) / y_true_clean[mask])) * 100
        
        # Directional Accuracy
        direction_true = np.diff(y_true_clean) > 0
        direction_pred = np.diff(y_pred_clean) > 0
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
        
        # Normalized RMSE (NRMSE)
        y_range = np.max(y_true_clean) - np.min(y_true_clean)
        nrmse = rmse / y_range if y_range > 0 else np.nan
        
        # ค่า Theil's U2 Statistic (ถ้า y_true ไม่เป็น 0)
        if np.sum(y_true_clean[:-1]**2) > 0:
            theil_u2 = np.sqrt(np.sum((y_pred_clean[1:] - y_true_clean[1:])**2) / 
                              np.sum((y_true_clean[1:] - y_true_clean[:-1])**2))
        else:
            theil_u2 = np.nan
        
        if verbose:
            print(f"\nStatistical Metrics:")
            print(f"  MSE: {mse:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  NRMSE: {nrmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  R²: {r2:.6f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
            print(f"  Theil's U2: {theil_u2:.6f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'nrmse': nrmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'theil_u2': theil_u2
        }
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                initial_balance: Optional[float] = None,
                                verbose: bool = True) -> Dict[str, Any]:
        """
        Calculate metrics related to trading performance
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            initial_balance: Initial trading balance
            verbose: Whether to print results to console
            
        Returns:
            Dictionary containing trading performance metrics
        """
        if initial_balance is None:
            initial_balance = self.config.INITIAL_BALANCE
        
        # ตรวจสอบข้อมูลนำเข้า
        if len(y_true) != len(y_pred):
            raise ValueError(f"Lengths of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) do not match")
        
        # ตรวจสอบและแก้ไข NaN, Inf
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # แจ้งเตือนถ้ามีการกรองข้อมูล
        if len(y_true_clean) < len(y_true):
            print(f"WARNING: Removed {len(y_true) - len(y_true_clean)} invalid values for trading simulation")
        
        # ค่าคอมมิชชัน
        commission = self.config.TRADING_COMMISSION
            
        # คำนวณทิศทางการเปลี่ยนแปลงของราคา
        price_changes = np.diff(y_true_clean)
        pred_changes = np.diff(y_pred_clean)
        
        # สร้างสัญญาณการซื้อขาย (1 สำหรับซื้อ, -1 สำหรับขาย)
        signals = np.sign(pred_changes)
        
        # คำนวณกำไร/ขาดทุนจากการเทรด (ต้องเลื่อนสัญญาณ 1 ช่วงเวลา)
        # เพราะเราไม่รู้ทิศทางจริงจนกว่าจะเห็นการทำนายถัดไป
        if len(signals) > 1:
            returns = signals[:-1] * price_changes[1:]
            signals_used = signals[:-1]  # สัญญาณที่ใช้จริง
        else:
            returns = np.array([])
            signals_used = np.array([])
        
        # ใส่ค่าคอมมิชชัน
        if len(returns) > 0:
            commission_cost = y_true_clean[1:-1] * commission
            returns = returns - commission_cost
            
            # คำนวณผลตอบแทนสะสม
            cumulative_returns = np.cumsum(returns)
            balance_curve = initial_balance + cumulative_returns
        else:
            cumulative_returns = np.array([0])
            balance_curve = np.array([initial_balance])
        
        # คำนวณอัตราผลตอบแทนประจำปี (Annualized Return)
        trading_days = len(returns)
        if trading_days > 0 and np.sum(np.abs(returns)) > 0:
            total_return = (balance_curve[-1] - initial_balance)
            annual_return = (total_return / initial_balance) * (252 / trading_days) * 100
        else:
            annual_return = 0.0
        
        # Win Rate
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Sharpe Ratio (using daily returns)
        daily_returns = returns / initial_balance
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        if len(balance_curve) > 1:
            peaks = np.maximum.accumulate(balance_curve)
            drawdowns = (peaks - balance_curve) / peaks * 100
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        else:
            drawdowns = np.array([0])
            max_drawdown = 0
        
        # Average Profit per Trade
        avg_profit = np.mean(returns) if len(returns) > 0 else 0
        avg_winning_trade = np.mean(returns[returns > 0]) if np.sum(returns > 0) > 0 else 0
        avg_losing_trade = np.mean(returns[returns < 0]) if np.sum(returns < 0) > 0 else 0
        
        # Risk-Reward Ratio
        risk_reward = abs(avg_winning_trade / avg_losing_trade) if avg_losing_trade != 0 else np.inf
        
        # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
        if total_trades > 0:
            win_percent = winning_trades / total_trades
            loss_percent = 1 - win_percent
            expectancy = (win_percent * avg_winning_trade) - (loss_percent * abs(avg_losing_trade))
        else:
            expectancy = 0
        
        if verbose:
            print(f"\nTrading Metrics:")
            print(f"  Annual Return: {annual_return:.2f}%")
            print(f"  Win Rate: {win_rate:.2f}%")
            print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {max_drawdown:.2f}%")
            print(f"  Total Trades: {total_trades}")
            print(f"  Winning Trades: {winning_trades} ({win_rate:.1f}%)")
            print(f"  Average Profit: {avg_profit:.6f}")
            print(f"  Risk-Reward Ratio: {risk_reward:.2f}")
            print(f"  Expectancy: {expectancy:.6f}")
            print(f"  Final Balance: {balance_curve[-1]:.2f} (from {initial_balance})")
        
        return {
            'annual_return': annual_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'avg_profit': avg_profit,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade, 
            'risk_reward_ratio': risk_reward,
            'expectancy': expectancy,
            'cumulative_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0,
            'final_balance': balance_curve[-1] if len(balance_curve) > 0 else initial_balance,
            'balance_curve': balance_curve.tolist(),
            'drawdown_curve': drawdowns.tolist(),
            'returns': returns.tolist() if len(returns) > 0 else [],
            'signals': signals_used.tolist() if len(signals_used) > 0 else []
        }
    
    def compare_with_benchmarks(self, y_true: np.ndarray, y_pred: np.ndarray,
                              verbose: bool = True) -> Dict[str, float]:
        """
        Compare model performance with benchmark strategies
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            verbose: Whether to print results to console
            
        Returns:
            Dictionary containing benchmark comparison metrics
        """
        # ตรวจสอบข้อมูลนำเข้า
        if len(y_true) != len(y_pred):
            raise ValueError(f"Lengths of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) do not match")
        
        # ตรวจสอบและแก้ไข NaN, Inf
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # แจ้งเตือนถ้ามีการกรองข้อมูล
        if len(y_true_clean) < len(y_true):
            print(f"WARNING: Removed {len(y_true) - len(y_true_clean)} invalid values for benchmark comparison")
        
        # กลยุทธ์ buy-and-hold
        buy_hold_return = (y_true_clean[-1] - y_true_clean[0]) / y_true_clean[0] * 100
        
        # กลยุทธ์ SMA Crossover (5 และ 20 วัน)
        sma5 = np.array([np.mean(y_true_clean[max(0, i-5):i]) if i > 0 else y_true_clean[0] for i in range(len(y_true_clean))])
        sma20 = np.array([np.mean(y_true_clean[max(0, i-20):i]) if i > 0 else y_true_clean[0] for i in range(len(y_true_clean))])
        
        # สร้างสัญญาณ (1 เมื่อ sma5 > sma20, -1 เมื่อ sma5 < sma20)
        sma_signals = np.sign(sma5 - sma20)
        
        # คำนวณผลตอบแทน
        price_changes = np.diff(y_true_clean)
        
        # สำหรับ SMA ผลตอบแทนเป็นผลคูณของสัญญาณและการเปลี่ยนแปลงราคา
        if len(sma_signals) > 1 and len(price_changes) > 0:
            sma_returns = np.sum(sma_signals[:-1] * price_changes)
            sma_return = sma_returns / y_true_clean[0] * 100
        else:
            sma_return = 0
        
        # กลยุทธ์สุ่ม
        np.random.seed(42)  # กำหนด seed เพื่อให้ผลลัพธ์เหมือนเดิมในทุกการรัน
        random_signals = np.random.choice([-1, 1], size=len(y_true_clean)-1)
        if len(random_signals) > 0 and len(price_changes) > 0:
            random_returns = np.sum(random_signals * price_changes)
            random_return = random_returns / y_true_clean[0] * 100
        else:
            random_return = 0
        
        # กลยุทธ์ของโมเดล
        if len(y_pred_clean) > 1:
            model_signals = np.sign(np.diff(y_pred_clean))
            if len(model_signals) > 1 and len(price_changes) > 1:
                model_returns = np.sum(model_signals[:-1] * price_changes[1:])
                model_return = model_returns / y_true_clean[0] * 100
            else:
                model_return = 0
        else:
            model_return = 0
        
        if verbose:
            print(f"\nBenchmark Comparison:")
            print(f"  Model Return: {model_return:.2f}%")
            print(f"  Buy & Hold Return: {buy_hold_return:.2f}%")
            print(f"  SMA Crossover Return: {sma_return:.2f}%")
            print(f"  Random Strategy Return: {random_return:.2f}%")
            
            # Calculate relative performance
            if buy_hold_return != 0:
                model_vs_buyhold = (model_return - buy_hold_return) / abs(buy_hold_return) * 100
                print(f"  Model outperformed Buy & Hold by: {model_vs_buyhold:.2f}%")
            
            if sma_return != 0:
                model_vs_sma = (model_return - sma_return) / abs(sma_return) * 100
                print(f"  Model outperformed SMA Crossover by: {model_vs_sma:.2f}%")
        
        return {
            'buy_hold_return': buy_hold_return,
            'sma_return': sma_return,
            'random_return': random_return,
            'model_return': model_return,
            'model_vs_buyhold': (model_return - buy_hold_return) / abs(buy_hold_return) * 100 if buy_hold_return != 0 else np.inf,
            'model_vs_sma': (model_return - sma_return) / abs(sma_return) * 100 if sma_return != 0 else np.inf
        }
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, dates: Optional[np.ndarray] = None, 
                     model_name: str = "model", pair: str = "unknown",
                     save_results: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a model with all metrics and optionally save results
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            dates: Optional array of dates for time-based analysis
            model_name: Name of the model for reporting
            pair: Currency pair being evaluated
            save_results: Whether to save results to file
            
        Returns:
            Dictionary containing all evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} on {pair}")
        print(f"{'='*60}")
        
        # ตรวจสอบข้อมูลนำเข้า
        if len(y_true) != len(y_pred):
            raise ValueError(f"Lengths of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) do not match")
        
        if dates is not None and len(dates) != len(y_true):
            print(f"WARNING: Dates array length ({len(dates)}) doesn't match data length ({len(y_true)})")
            dates = None
        
        # คำนวณเมตริกต่างๆ
        metrics = self.calculate_metrics(y_true, y_pred)
        trading_metrics = self.calculate_trading_metrics(y_true, y_pred)
        benchmark_metrics = self.compare_with_benchmarks(y_true, y_pred)
        
        # รวมผลลัพธ์ทั้งหมด
        results = {
            'info': {
                'model_name': model_name,
                'pair': pair,
                'data_points': len(y_true),
                'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'metrics': metrics,
            'trading_metrics': trading_metrics,
            'benchmark': benchmark_metrics
        }
        
        # เพิ่มการวิเคราะห์ตามช่วงเวลาถ้ามีข้อมูลวันที่
        if dates is not None:
            time_analysis = self._analyze_by_time_period(y_true, y_pred, dates)
            results['time_analysis'] = time_analysis
        
        # บันทึกผลลัพธ์
        if save_results:
            self._save_evaluation_results(results, model_name, pair)
        
        return results
    
    def _analyze_by_time_period(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             dates: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Analyze model performance by different time periods
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            dates: Array of dates corresponding to data points
            
        Returns:
            Dictionary with metrics for different time periods
        """
        # แปลง dates เป็น pandas datetime ถ้ายังไม่ใช่
        if not isinstance(dates[0], pd.Timestamp):
            dates = pd.to_datetime(dates)
        
        # สร้าง DataFrame เพื่อง่ายต่อการวิเคราะห์
        df = pd.DataFrame({
            'date': dates,
            'y_true': y_true,
            'y_pred': y_pred
        })
        
        # เพิ่มคอลัมน์ที่ใช้ในการวิเคราะห์
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['hour'] = df['date'].dt.hour
        
        # คำนวณความแม่นยำตามช่วงเวลาต่างๆ
        results = {}
        
        # วิเคราะห์ตามปี
        year_groups = df.groupby('year')
        results['by_year'] = {}
        
        for year, group in year_groups:
            if len(group) > 1:  # ต้องมีข้อมูลอย่างน้อย 2 จุดเพื่อคำนวณ directional accuracy
                metrics = self.calculate_metrics(group['y_true'].values, group['y_pred'].values, verbose=False)
                results['by_year'][str(year)] = {
                    'rmse': metrics['rmse'],
                    'directional_accuracy': metrics['directional_accuracy']
                }
        
        # วิเคราะห์ตามเดือน
        month_groups = df.groupby('month')
        results['by_month'] = {}
        
        for month, group in month_groups:
            if len(group) > 1:
                metrics = self.calculate_metrics(group['y_true'].values, group['y_pred'].values, verbose=False)
                results['by_month'][str(month)] = {
                    'rmse': metrics['rmse'],
                    'directional_accuracy': metrics['directional_accuracy']
                }
        
        # วิเคราะห์ตามวันในสัปดาห์
        day_groups = df.groupby('day_of_week')
        results['by_day_of_week'] = {}
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day, group in day_groups:
            if len(group) > 1:
                metrics = self.calculate_metrics(group['y_true'].values, group['y_pred'].values, verbose=False)
                results['by_day_of_week'][day_names[day]] = {
                    'rmse': metrics['rmse'],
                    'directional_accuracy': metrics['directional_accuracy']
                }
        
        # วิเคราะห์ตามชั่วโมง
        hour_groups = df.groupby('hour')
        results['by_hour'] = {}
        
        for hour, group in hour_groups:
            if len(group) > 1:
                metrics = self.calculate_metrics(group['y_true'].values, group['y_pred'].values, verbose=False)
                results['by_hour'][str(hour)] = {
                    'rmse': metrics['rmse'],
                    'directional_accuracy': metrics['directional_accuracy']
                }
        
        return results
    
    def _save_evaluation_results(self, results: Dict[str, Any], model_name: str, pair: str) -> None:
        """
        Save evaluation results to file
        
        Args:
            results: Dictionary containing evaluation results
            model_name: Name of the model
            pair: Currency pair being evaluated
        """
        # สร้างชื่อไฟล์
        filename = f"{pair}_{model_name}_evaluation.json"
        filepath = os.path.join(self.results_path, filename)
        
        # แปลง NumPy arrays เป็น lists เพื่อให้สามารถบันทึกเป็น JSON ได้
        results_json = self._prepare_results_for_json(results)
        
        # บันทึกผลลัพธ์
        try:
            with open(filepath, 'w') as f:
                json.dump(results_json, f, indent=4)
            print(f"\nEvaluation results saved to {filepath}")
        except Exception as e:
            print(f"Error saving evaluation results: {e}")
    
    def _prepare_results_for_json(self, obj: Any) -> Any:
        """
        Convert NumPy types to Python native types for JSON serialization
        
        Args:
            obj: Object to convert
            
        Returns:
            Converted object suitable for JSON serialization
        """
        if isinstance(obj, dict):
            return {k: self._prepare_results_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_results_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif obj is np.nan:
            return None
        elif obj is np.inf:
            return "Infinity"
        elif obj is -np.inf:
            return "-Infinity"
        else:
            return obj
    
    def forward_walking_validation(self, model, X, y, window_size=500, step=100):
        """
        Perform forward walking validation
        
        Args:
            model: Model with fit and predict methods
            X: Features array
            y: Target array
            window_size: Size of the rolling training window
            step: Number of steps to move forward in each iteration
            
        Returns:
            Tuple of (predictions, actuals)
        """
        if len(X) <= window_size:
            raise ValueError(f"Not enough samples ({len(X)}) for window_size ({window_size})")
            
        predictions = []
        actuals = []
        
        for i in range(0, len(X) - window_size, step):
            # ฝึกฝนบนหน้าต่าง
            X_train = X[i:i+window_size]
            y_train = y[i:i+window_size]
            
            # ทำนายจุดถัดไป
            X_test = X[i+window_size:i+window_size+step]
            y_test = y[i+window_size:i+window_size+step]
            
            if len(X_test) == 0:
                continue
                
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            predictions.extend(preds)
            actuals.extend(y_test)
        
        return np.array(predictions), np.array(actuals)
    
    def directional_accuracy_at_key_levels(self, y_true, y_pred, prices, threshold=0.01):
        """
        Calculate directional accuracy at key support/resistance levels
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            prices: Series or array of price data
            threshold: Threshold for identifying key levels (as percentage)
            
        Returns:
            Directional accuracy at key levels as a percentage
        """
        # ตรวจสอบรูปแบบของข้อมูล
        if isinstance(prices, pd.Series):
            prices_series = prices
        else:
            prices_series = pd.Series(prices)
        
        # ระบุระดับสำคัญ (วิธีอย่างง่าย)
        highs = prices_series.rolling(20).max()
        lows = prices_series.rolling(20).min()
        
        # ตรวจสอบว่าราคาอยู่ใกล้ระดับสำคัญหรือไม่ (ภายในเกณฑ์)
        near_key_level = (
            (np.abs(prices_series - highs) / prices_series < threshold) | 
            (np.abs(prices_series - lows) / prices_series < threshold)
        )
        
        # คำนวณความแม่นยำในการทำนายทิศทางเฉพาะที่จุดเหล่านี้
        if sum(near_key_level) > 0:
            direction_true = np.sign(np.diff(y_true))
            direction_pred = np.sign(np.diff(y_pred))
            
            # กรองสำหรับระดับสำคัญ
            key_level_idx = near_key_level[:-1]  # ปรับสำหรับความยาว diff()
            
            if sum(key_level_idx) > 0:
                key_level_accuracy = np.mean(
                    direction_true[key_level_idx] == direction_pred[key_level_idx]
                ) * 100
                
                return key_level_accuracy
        
        return None