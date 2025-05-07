"""
Evaluation Utilities for Forex Prediction
This module contains helper functions for evaluating forex prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
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
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
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
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
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
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        return np.mean(direction_true == direction_pred) * 100
    
    @staticmethod
    def calculate_trading_returns(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        initial_balance: float = 10000, 
        position_size: float = 1.0,
        commission: float = 0.0001
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Calculate returns from a simple trading strategy based on predicted price movements
        
        Args:
            y_true: Array of true prices
            y_pred: Array of predicted prices
            initial_balance: Initial account balance
            position_size: Fraction of balance to risk per trade (0-1)
            commission: Trading commission as fraction of trade size
            
        Returns:
            Tuple of (returns_array, balance_curve, metrics_dict)
        """
        # Calculate price changes and predicted directions
        price_changes = np.diff(y_true)
        pred_directions = np.sign(np.diff(y_pred))
        
        # Skip the first element since we can't trade based on it
        tradable_changes = price_changes[1:]
        tradable_directions = pred_directions[:-1]
        
        # Calculate trade returns (without compounding)
        trade_returns = tradable_directions * tradable_changes
        
        # Apply commission costs
        commission_costs = np.abs(y_true[:-2]) * commission
        trade_returns = trade_returns - commission_costs
        
        # Calculate cumulative returns and balance curve
        position_value = initial_balance * position_size
        trade_values = trade_returns * position_value / y_true[:-2]  # Convert price change to percentage
        cumulative_returns = np.cumsum(trade_values)
        balance_curve = initial_balance + cumulative_returns
        
        # Calculate trading metrics
        metrics = {}
        
        # Total return
        total_return_pct = (balance_curve[-1] - initial_balance) / initial_balance * 100
        metrics['total_return_pct'] = total_return_pct
        
        # Annualized return (assuming 252 trading days per year)
        n_days = len(tradable_changes)
        annualized_return = ((balance_curve[-1] / initial_balance) ** (252 / n_days) - 1) * 100
        metrics['annualized_return'] = annualized_return
        
        # Win rate
        winning_trades = np.sum(trade_values > 0)
        total_trades = len(trade_values)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        metrics['win_rate'] = win_rate
        
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
        metrics['risk_reward'] = risk_reward
        
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
        # Calculate total return
        start_price = y_true[0]
        end_price = y_true[-1]
        total_return_pct = (end_price - start_price) / start_price * 100
        
        # Calculate final balance
        final_balance = initial_balance * (1 + total_return_pct / 100)
        
        # Calculate annualized return (assuming 252 trading days per year)
        n_days = len(y_true)
        annualized_return = ((end_price / start_price) ** (252 / n_days) - 1) * 100
        
        # Maximum drawdown
        peak = np.maximum.accumulate(y_true)
        drawdown = (peak - y_true) / peak * 100
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
        annualized_return = ((balance_curve[-1] / initial_balance) ** (252 / n_days) - 1) * 100
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
        
        return trade_returns, balance_curve, metrics