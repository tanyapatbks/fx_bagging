"""
Evaluation Utilities for Forex Prediction
This module contains helper functions for evaluating forex prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ForexEvaluationMetrics:
    """
    Helper class for calculating various metrics for forex prediction evaluation
    """
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            
        Returns:
            RMSE value
        """
        # ตรวจสอบว่า arrays มีขนาดเท่ากัน
        if len(y_true) != len(y_pred):
            raise ValueError(f"Lengths of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) do not match")
            
        # ตรวจสอบ NaN และ Infinity
        if np.isnan(y_true).any() or np.isnan(y_pred).any() or np.isinf(y_true).any() or np.isinf(y_pred).any():
            print("WARNING: NaN or Inf values detected in input arrays. Removing them for calculation.")
            mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            # ตรวจสอบว่าเหลือข้อมูลเพียงพอ
            if len(y_true) == 0:
                return float('nan')
        
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            
        Returns:
            MAPE value as a percentage
        """
        # ตรวจสอบว่า arrays มีขนาดเท่ากัน
        if len(y_true) != len(y_pred):
            raise ValueError(f"Lengths of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) do not match")
            
        # ตรวจสอบ NaN และ Infinity
        if np.isnan(y_true).any() or np.isnan(y_pred).any() or np.isinf(y_true).any() or np.isinf(y_pred).any():
            print("WARNING: NaN or Inf values detected in input arrays. Removing them for calculation.")
            mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            # ตรวจสอบว่าเหลือข้อมูลเพียงพอ
            if len(y_true) == 0:
                return float('nan')
                
        # Avoid division by zero
        mask = y_true != 0
        if np.sum(mask) == 0:
            return float('nan')
            
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Directional Accuracy
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            
        Returns:
            Directional accuracy as a percentage
        """
        # ตรวจสอบว่า arrays มีขนาดเท่ากัน
        if len(y_true) != len(y_pred):
            raise ValueError(f"Lengths of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) do not match")
            
        # ตรวจสอบว่ามีข้อมูลเพียงพอ
        if len(y_true) <= 1:
            raise ValueError("Not enough data points to calculate directional accuracy")
            
        # ตรวจสอบ NaN และ Infinity
        if np.isnan(y_true).any() or np.isnan(y_pred).any() or np.isinf(y_true).any() or np.isinf(y_pred).any():
            print("WARNING: NaN or Inf values detected in input arrays. Removing them for calculation.")
            
            # สร้าง arrays ใหม่โดยลบค่า NaN และ Infinity
            valid_indices = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
            y_true_clean = y_true[valid_indices]
            y_pred_clean = y_pred[valid_indices]
            
            # ตรวจสอบว่าเหลือข้อมูลเพียงพอ
            if len(y_true_clean) <= 1:
                return float('nan')
                
            # คำนวณทิศทาง
            direction_true = np.diff(y_true_clean) > 0
            direction_pred = np.diff(y_pred_clean) > 0
        else:
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            
        return float(np.mean(direction_true == direction_pred) * 100)
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            
        Returns:
            Dictionary containing various evaluation metrics
        """
        metrics = {}
        
        # ตรวจสอบข้อมูลนำเข้า
        if len(y_true) != len(y_pred):
            raise ValueError(f"Lengths of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) do not match")
            
        # ตรวจสอบ NaN และ Infinity
        if np.isnan(y_true).any() or np.isnan(y_pred).any() or np.isinf(y_true).any() or np.isinf(y_pred).any():
            print("WARNING: NaN or Inf values detected in input arrays. Cleaning data for calculation.")
            mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            # ตรวจสอบว่าเหลือข้อมูลเพียงพอ
            if len(y_true_clean) == 0:
                return {metric: float('nan') for metric in 
                       ['mse', 'rmse', 'mae', 'r2', 'mape', 'directional_accuracy']}
        else:
            y_true_clean = y_true
            y_pred_clean = y_pred
            
        # สถิติพื้นฐาน
        try:
            metrics['mse'] = float(mean_squared_error(y_true_clean, y_pred_clean))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(y_true_clean, y_pred_clean))
            metrics['r2'] = float(r2_score(y_true_clean, y_pred_clean))
        except Exception as e:
            print(f"Error calculating basic metrics: {e}")
            metrics['mse'] = metrics['rmse'] = metrics['mae'] = metrics['r2'] = float('nan')
        
        # Mean Absolute Percentage Error (MAPE)
        try:
            # Avoid division by zero
            mask = y_true_clean != 0
            if np.sum(mask) > 0:
                metrics['mape'] = float(np.mean(np.abs((y_true_clean[mask] - y_pred_clean[mask]) / y_true_clean[mask])) * 100)
            else:
                metrics['mape'] = float('nan')
        except Exception as e:
            print(f"Error calculating MAPE: {e}")
            metrics['mape'] = float('nan')
        
        # Directional Accuracy
        try:
            if len(y_true_clean) > 1:
                direction_true = np.diff(y_true_clean) > 0
                direction_pred = np.diff(y_pred_clean) > 0
                metrics['directional_accuracy'] = float(np.mean(direction_true == direction_pred) * 100)
            else:
                metrics['directional_accuracy'] = float('nan')
        except Exception as e:
            print(f"Error calculating directional accuracy: {e}")
            metrics['directional_accuracy'] = float('nan')
        
        return metrics
    
    @staticmethod
    def calculate_trading_returns(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        initial_balance: float = 10000, 
        position_size: float = 1.0,
        commission: float = 0.0001,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Calculate returns from a simple trading strategy based on predicted price movements
        
        Args:
            y_true: Array of true prices
            y_pred: Array of predicted prices
            initial_balance: Initial account balance
            position_size: Fraction of balance to risk per trade (0-1)
            commission: Trading commission as fraction of trade size
            stop_loss: Optional stop loss as percentage (e.g., 0.01 for 1%)
            take_profit: Optional take profit as percentage (e.g., 0.02 for 2%)
            
        Returns:
            Tuple of (returns_array, balance_curve, metrics_dict)
        """
        # ตรวจสอบข้อมูลนำเข้า
        if len(y_true) != len(y_pred):
            raise ValueError(f"Lengths of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) do not match")
            
        if len(y_true) < 3:
            raise ValueError("Not enough data points for trading simulation (minimum 3 required)")
            
        # ตรวจสอบ NaN และ Infinity
        if np.isnan(y_true).any() or np.isnan(y_pred).any() or np.isinf(y_true).any() or np.isinf(y_pred).any():
            print("WARNING: NaN or Inf values detected in input arrays. Cleaning data for simulation.")
            mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            # ตรวจสอบว่าเหลือข้อมูลเพียงพอ
            if len(y_true) < 3:
                empty_result = np.array([]), np.array([initial_balance]), {
                    'total_return_pct': 0.0,
                    'annualized_return': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'avg_trade': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'risk_reward_ratio': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0
                }
                return empty_result
        
        # Calculate price changes and predicted directions
        price_changes = np.diff(y_true)
        pred_directions = np.sign(np.diff(y_pred))
        
        # Skip the first element since we can't trade based on it
        tradable_changes = price_changes[1:]
        tradable_directions = pred_directions[:-1]
        
        # Validate arrays
        if len(tradable_changes) != len(tradable_directions):
            raise ValueError(f"Mismatch in tradable arrays: changes ({len(tradable_changes)}), directions ({len(tradable_directions)})")
            
        # Calculate trade returns (without compounding)
        trade_returns = tradable_directions * tradable_changes
        
        # Apply commission costs
        commission_costs = np.abs(y_true[:-2]) * commission
        trade_returns = trade_returns - commission_costs
        
        # Calculate cumulative returns and balance curve
        position_value = initial_balance * position_size
        trade_values = trade_returns * position_value / y_true[:-2]  # Convert price change to percentage
        
        # Apply stop loss and take profit if provided
        if stop_loss is not None or take_profit is not None:
            for i in range(len(trade_values)):
                pct_change = trade_values[i] / position_value
                
                # Apply stop loss
                if stop_loss is not None and pct_change < -stop_loss:
                    trade_values[i] = -stop_loss * position_value
                
                # Apply take profit
                if take_profit is not None and pct_change > take_profit:
                    trade_values[i] = take_profit * position_value
        
        cumulative_returns = np.cumsum(trade_values)
        balance_curve = initial_balance + cumulative_returns
        
        # Calculate trading metrics
        metrics = {}
        
        # Total return
        total_return_pct = (balance_curve[-1] - initial_balance) / initial_balance * 100
        metrics['total_return_pct'] = total_return_pct
        
        # Annualized return (assuming 252 trading days per year)
        n_days = len(tradable_changes)
        if n_days > 0 and balance_curve[-1] > 0:
            annualized_return = ((balance_curve[-1] / initial_balance) ** (252 / n_days) - 1) * 100
        else:
            annualized_return = 0.0
        metrics['annualized_return'] = annualized_return
        
        # Win rate
        winning_trades = np.sum(trade_values > 0)
        total_trades = len(trade_values)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        metrics['win_rate'] = win_rate
        metrics['total_trades'] = total_trades
        metrics['winning_trades'] = winning_trades
        
        # Profit factor
        gross_profit = np.sum(trade_values[trade_values > 0])
        gross_loss = np.abs(np.sum(trade_values[trade_values < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        metrics['profit_factor'] = profit_factor
        
        # Sharpe ratio (assuming risk-free rate of 0)
        daily_returns = trade_values / position_value
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # Maximum drawdown
        peak = np.maximum.accumulate(balance_curve)
        drawdown = (peak - balance_curve) / peak * 100
        max_drawdown = np.max(drawdown)
        metrics['max_drawdown'] = max_drawdown
        
        # Average trade
        avg_trade = np.mean(trade_values)
        metrics['avg_trade'] = avg_trade
        
        # Average win/loss
        avg_win = np.mean(trade_values[trade_values > 0]) if any(trade_values > 0) else 0
        avg_loss = np.mean(trade_values[trade_values < 0]) if any(trade_values < 0) else 0
        metrics['avg_win'] = avg_win
        metrics['avg_loss'] = avg_loss
        
        # Risk-reward ratio
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        metrics['risk_reward_ratio'] = risk_reward
        
        # ข้อมูลการซื้อขายเพิ่มเติม
        metrics['balance_curve'] = balance_curve
        metrics['drawdown_curve'] = drawdown
        metrics['returns'] = trade_returns
        metrics['signals'] = tradable_directions
        
        return trade_returns, balance_curve, metrics
    
    @staticmethod
    def calculate_holding_returns(
        y_true: np.ndarray, 
        initial_balance: float = 10000
    ) -> Dict[str, float]:
        """
        Calculate returns from a buy-and-hold strategy
        
        Args:
            y_true: Array of true prices
            initial_balance: Initial account balance
            
        Returns:
            Dictionary of buy-and-hold metrics
        """
        # ตรวจสอบข้อมูลนำเข้า
        if len(y_true) < 2:
            raise ValueError("Not enough data points for buy-and-hold simulation (minimum 2 required)")
            
        # ตรวจสอบ NaN และ Infinity
        if np.isnan(y_true).any() or np.isinf(y_true).any():
            print("WARNING: NaN or Inf values detected in y_true. Cleaning data for simulation.")
            y_true = y_true[~np.isnan(y_true) & ~np.isinf(y_true)]
            
            # ตรวจสอบว่าเหลือข้อมูลเพียงพอ
            if len(y_true) < 2:
                return {
                    'buy_hold_return': 0.0,
                    'buy_hold_final_balance': initial_balance,
                    'buy_hold_annualized_return': 0.0,
                    'buy_hold_max_drawdown': 0.0
                }
                
        # Calculate total return
        start_price = y_true[0]
        end_price = y_true[-1]
        
        # ป้องกันการหารด้วย 0
        if start_price == 0:
            print("WARNING: Starting price is zero. Using small positive value.")
            start_price = 0.001
            
        total_return_pct = (end_price - start_price) / start_price * 100
        
        # Calculate final balance
        final_balance = initial_balance * (1 + total_return_pct / 100)
        
        # Calculate annualized return (assuming 252 trading days per year)
        n_days = len(y_true)
        if end_price > 0 and start_price > 0:
            annualized_return = ((end_price / start_price) ** (252 / n_days) - 1) * 100
        else:
            annualized_return = 0.0
        
        # Maximum drawdown
        balance_curve = initial_balance * (y_true / start_price)
        peak = np.maximum.accumulate(balance_curve)
        drawdown = (peak - balance_curve) / peak * 100
        max_drawdown = np.max(drawdown)
        
        return {
            'buy_hold_return': total_return_pct,
            'buy_hold_final_balance': final_balance,
            'buy_hold_annualized_return': annualized_return,
            'buy_hold_max_drawdown': max_drawdown
        }
    
    @staticmethod
    def calculate_sma_crossover_returns(
        y_true: np.ndarray, 
        initial_balance: float = 10000,
        short_period: int = 5,
        long_period: int = 20,
        position_size: float = 1.0,
        commission: float = 0.0001
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Calculate returns from a SMA crossover strategy
        
        Args:
            y_true: Array of true prices
            initial_balance: Initial account balance
            short_period: Period for short SMA
            long_period: Period for long SMA
            position_size: Fraction of balance to risk per trade (0-1)
            commission: Trading commission as fraction of trade size
            
        Returns:
            Tuple of (returns_array, balance_curve, metrics_dict)
        """
        # ตรวจสอบข้อมูลนำเข้า
        if len(y_true) <= long_period:
            raise ValueError(f"Not enough data points for SMA calculation (minimum {long_period+1} required)")
            
        # ตรวจสอบ NaN และ Infinity
        if np.isnan(y_true).any() or np.isinf(y_true).any():
            print("WARNING: NaN or Inf values detected in y_true. Cleaning data for simulation.")
            # เราไม่สามารถลบค่า NaN และ Infinity ได้โดยตรง เพราะจะทำให้ลำดับเวลาเปลี่ยนไป
            # แทนที่ด้วยค่าที่ถูกต้อง
            y_true = np.nan_to_num(y_true, nan=np.nanmean(y_true), posinf=np.nanmax(y_true), neginf=np.nanmin(y_true))
        
        # Calculate SMAs
        sma_short = np.zeros_like(y_true)
        sma_long = np.zeros_like(y_true)
        
        for i in range(len(y_true)):
            if i >= short_period:
                sma_short[i] = np.mean(y_true[i-short_period:i])
            else:
                sma_short[i] = y_true[i]
                
            if i >= long_period:
                sma_long[i] = np.mean(y_true[i-long_period:i])
            else:
                sma_long[i] = y_true[i]
        
        # Generate signals (1 for long, -1 for short)
        signals = np.sign(sma_short - sma_long)
        
        # Calculate price changes
        price_changes = np.diff(y_true)
        
        # Align signals with price changes (signals from previous day applied to next day's change)
        trading_signals = signals[:-1]
        
        # Calculate trade returns
        trade_returns = trading_signals * price_changes
        
        # Apply commission costs
        commission_costs = np.abs(y_true[:-1]) * commission
        trade_returns = trade_returns - commission_costs
        
        # Calculate cumulative returns and balance curve
        position_value = initial_balance * position_size
        trade_values = trade_returns * position_value / y_true[:-1]  # Convert price change to percentage
        cumulative_returns = np.cumsum(trade_values)
        balance_curve = initial_balance + cumulative_returns
        
        # Calculate trading metrics
        metrics = {}
        
        # Total return
        total_return_pct = (balance_curve[-1] - initial_balance) / initial_balance * 100
        metrics['sma_return'] = total_return_pct
        
        # Annualized return (assuming 252 trading days per year)
        n_days = len(price_changes)
        if n_days > 0 and balance_curve[-1] > 0:
            annualized_return = ((balance_curve[-1] / initial_balance) ** (252 / n_days) - 1) * 100
        else:
            annualized_return = 0.0
        metrics['sma_annualized_return'] = annualized_return
        
        # Win rate
        winning_trades = np.sum(trade_values > 0)
        total_trades = len(trade_values)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        metrics['sma_win_rate'] = win_rate
        
        # Maximum drawdown
        peak = np.maximum.accumulate(balance_curve)
        drawdown = (peak - balance_curve) / peak * 100
        max_drawdown = np.max(drawdown)
        metrics['sma_max_drawdown'] = max_drawdown
        
        # Signal changes (number of trades)
        signal_changes = np.sum(np.abs(np.diff(trading_signals)) > 0) + 1  # +1 for initial position
        metrics['sma_trades'] = signal_changes
        
        return trade_returns, balance_curve, metrics
    
    @staticmethod
    def calculate_random_returns(
        y_true: np.ndarray, 
        initial_balance: float = 10000,
        position_size: float = 1.0,
        commission: float = 0.0001,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Calculate returns from a random strategy (baseline)
        
        Args:
            y_true: Array of true prices
            initial_balance: Initial account balance
            position_size: Fraction of balance to risk per trade (0-1)
            commission: Trading commission as fraction of trade size
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (returns_array, balance_curve, metrics_dict)
        """
        # ตรวจสอบข้อมูลนำเข้า
        if len(y_true) < 2:
            raise ValueError("Not enough data points for random strategy simulation (minimum 2 required)")
            
        # ตรวจสอบ NaN และ Infinity
        if np.isnan(y_true).any() or np.isinf(y_true).any():
            print("WARNING: NaN or Inf values detected in y_true. Cleaning data for simulation.")
            mask = ~(np.isnan(y_true) | np.isinf(y_true))
            y_true_clean = y_true[mask]
            
            # ตรวจสอบว่าเหลือข้อมูลเพียงพอ
            if len(y_true_clean) < 2:
                empty_result = np.array([]), np.array([initial_balance]), {
                    'random_return': 0.0,
                    'random_annualized_return': 0.0,
                    'random_win_rate': 0.0,
                    'random_max_drawdown': 0.0,
                    'random_trades': 0
                }
                return empty_result
                
            # ใช้ข้อมูลที่สะอาด
            y_true = y_true_clean
        
        # Set random seed
        np.random.seed(seed)
        
        # Generate random signals (1 for long, -1 for short)
        signals = np.random.choice([-1, 1], size=len(y_true))
        
        # Calculate price changes
        price_changes = np.diff(y_true)
        
        # Align signals with price changes (signals from previous day applied to next day's change)
        trading_signals = signals[:-1]
        
        # Calculate trade returns
        trade_returns = trading_signals * price_changes
        
        # Apply commission costs
        commission_costs = np.abs(y_true[:-1]) * commission
        trade_returns = trade_returns - commission_costs
        
        # Calculate cumulative returns and balance curve
        position_value = initial_balance * position_size
        trade_values = trade_returns * position_value / y_true[:-1]  # Convert price change to percentage
        cumulative_returns = np.cumsum(trade_values)
        balance_curve = initial_balance + cumulative_returns
        
        # Calculate trading metrics
        metrics = {}
        
        # Total return
        total_return_pct = (balance_curve[-1] - initial_balance) / initial_balance * 100
        metrics['random_return'] = total_return_pct
        
        # Annualized return (assuming 252 trading days per year)
        n_days = len(price_changes)
        if n_days > 0 and balance_curve[-1] > 0:
            annualized_return = ((balance_curve[-1] / initial_balance) ** (252 / n_days) - 1) * 100
        else:
            annualized_return = 0.0
        metrics['random_annualized_return'] = annualized_return
        
        # Win rate
        winning_trades = np.sum(trade_values > 0)
        total_trades = len(trade_values)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        metrics['random_win_rate'] = win_rate
        
        # Maximum drawdown
        peak = np.maximum.accumulate(balance_curve)
        drawdown = (peak - balance_curve) / peak * 100
        max_drawdown = np.max(drawdown)
        metrics['random_max_drawdown'] = max_drawdown
        
        # Signal changes (number of trades)
        signal_changes = np.sum(np.abs(np.diff(trading_signals)) > 0) + 1  # +1 for initial position
        metrics['random_trades'] = signal_changes
        
        return trade_returns, balance_curve, metrics
    
    @staticmethod
    def forward_walking_validation(model, X, y, window_size=500, step=100):
        """
        Perform forward walking validation (a.k.a. walk-forward optimization)
        
        Args:
            model: Model object with fit and predict methods
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
            
            # ตรวจสอบว่ามีข้อมูลทดสอบหรือไม่
            if len(X_test) == 0:
                continue
                
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            predictions.extend(preds)
            actuals.extend(y_test)
        
        return np.array(predictions), np.array(actuals)
    
    @staticmethod
    def directional_accuracy_at_key_levels(y_true, y_pred, prices, threshold=0.01):
        """
        Calculate directional accuracy at key support/resistance levels
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            prices: Series of price data
            threshold: Threshold for identifying key levels (as percentage)
            
        Returns:
            Directional accuracy at key levels as a percentage
        """
        # ตรวจสอบข้อมูลนำเข้า
        if len(y_true) != len(y_pred):
            raise ValueError(f"Lengths of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) do not match")
            
        if len(prices) < 20:
            raise ValueError("Not enough price data to identify key levels (minimum 20 required)")
            
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
            
            # กรองสำหรับระดับสำคัญ (ปรับความยาวของ near_key_level ให้ตรงกับ diff)
            key_level_idx = near_key_level[:-1]  # ปรับสำหรับความยาว diff()
            
            # ตรวจสอบว่ามีข้อมูลเพียงพอ
            if sum(key_level_idx) == 0:
                return None
                
            # คำนวณความแม่นยำเฉพาะที่ระดับสำคัญ
            key_level_accuracy = np.mean(
                direction_true[key_level_idx] == direction_pred[key_level_idx]
            ) * 100
            
            return key_level_accuracy
        return None