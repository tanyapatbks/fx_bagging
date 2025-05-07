"""
Visualization Module for Forex Prediction
This module handles the creation of visualizations for results analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, Any, List


class ResultVisualizer:
    """
    Class for visualizing prediction results
    """
    
    def __init__(self, config):
        """
        Initialize the ResultVisualizer with configuration parameters
        
        Args:
            config: Configuration object containing paths and parameters
        """
        self.config = config
        
        # Set style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.config.RESULTS_PATH):
            os.makedirs(self.config.RESULTS_PATH)
    
    def plot_predictions_comparison(self, model_results: Dict[str, Dict[str, Any]], 
                                 model_identifier: str, pair: str) -> None:
        """
        Plot comparison of predictions from different models
        
        Args:
            model_results: Dictionary containing results for different models
            model_identifier: Identifier for the model (e.g., 'E1', 'G2', 'B')
            pair: Currency pair being visualized
        """
        plt.figure(figsize=(16, 8))
        
        # Get actual values from any model since they are all the same
        actual = model_results['LSTM']['actual']
        dates = model_results['LSTM']['dates']
        
        # Plot actual values
        plt.plot(dates, actual, label='Actual', color='black', linewidth=2)
        
        # Plot predictions for each model
        colors = {'LSTM': 'royalblue', 'GRU': 'darkorange', 'XGBoost': 'forestgreen', 'TFT': 'crimson'}
        for model_name, color in colors.items():
            if model_name in model_results:
                predicted = model_results[model_name]['predictions']
                plt.plot(dates, predicted, label=f'{model_name} Prediction', color=color, alpha=0.7)
        
        # Set title and labels
        plt.title(f'{pair} - {model_identifier} Models Predictions Comparison')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        
        # Add annotation with date range
        plt.annotate(f'Test Period: {self.config.TEST_START} to {self.config.TEST_END}', 
                   xy=(0.02, 0.02), xycoords='figure fraction', 
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, f"{pair}_{model_identifier}_predictions.png"), dpi=300)
        plt.close()
    
    def plot_metrics_comparison(self, model_evaluation: Dict[str, Dict[str, Any]], 
                             model_identifier: str, pair: str) -> None:
        """
        Plot comparison of evaluation metrics for different models
        
        Args:
            model_evaluation: Dictionary containing evaluation metrics for different models
            model_identifier: Identifier for the model (e.g., 'E1', 'G2', 'B')
            pair: Currency pair being visualized
        """
        # กำหนดสี
        colors = {'LSTM': 'royalblue', 'GRU': 'darkorange', 'XGBoost': 'forestgreen', 'TFT': 'crimson'}
        
        # 1. แสดงกราฟเปรียบเทียบ RMSE, MAE, MAPE
        plt.figure(figsize=(18, 6))
        
        metrics_names = ['RMSE', 'MAE', 'MAPE (%)']
        metrics_keys = ['rmse', 'mae', 'mape']
        
        for i, (name, key) in enumerate(zip(metrics_names, metrics_keys)):
            plt.subplot(1, 3, i+1)
            
            models = list(model_evaluation.keys())
            values = [model_evaluation[model]['metrics'][key] for model in models]
            
            bars = plt.bar(models, values, color=[colors[model] for model in models], alpha=0.7)
            
            # เพิ่มค่าบนแท่งกราฟ
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, value + max(values) * 0.02, 
                        f'{value:.4f}' if key != 'mape' else f'{value:.2f}', 
                        ha='center', va='bottom', fontweight='bold')
            
            plt.title(f'{pair} - {model_identifier} - {name} Comparison')
            plt.ylabel(name)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, f"{pair}_{model_identifier}_statistical_metrics.png"), dpi=300)
        plt.close()
        
        # 2. แสดงกราฟเปรียบเทียบ Annual Return, Win Rate, Sharpe Ratio
        plt.figure(figsize=(18, 6))
        
        metrics_names = ['Annual Return (%)', 'Win Rate (%)', 'Sharpe Ratio']
        metrics_keys = ['annual_return', 'win_rate', 'sharpe_ratio']
        
        for i, (name, key) in enumerate(zip(metrics_names, metrics_keys)):
            plt.subplot(1, 3, i+1)
            
            models = list(model_evaluation.keys())
            values = [model_evaluation[model]['trading_metrics'][key] for model in models]
            
            bars = plt.bar(models, values, color=[colors[model] for model in models], alpha=0.7)
            
            # เพิ่มค่าบนแท่งกราฟ
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, value + max(abs(min(values)), abs(max(values))) * 0.05, 
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.title(f'{pair} - {model_identifier} - {name} Comparison')
            plt.ylabel(name)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, f"{pair}_{model_identifier}_trading_metrics.png"), dpi=300)
        plt.close()
        
        # 3. แสดงกราฟเปรียบเทียบ Returns with Benchmarks
        plt.figure(figsize=(14, 8))
        
        benchmark_types = ['Model Return', 'Buy & Hold', 'SMA Crossover', 'Random']
        benchmark_keys = ['model_return', 'buy_hold_return', 'sma_return', 'random_return']
        models = list(model_evaluation.keys())
        
        x = np.arange(len(benchmark_types))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            offset = (i - len(models) / 2 + 0.5) * width
            values = [model_evaluation[model]['benchmark'][key] for key in benchmark_keys]
            
            plt.bar(x + offset, values, width, label=model, color=colors[model], alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f'{pair} - {model_identifier} - Return Comparison with Benchmarks')
        plt.ylabel('Return (%)')
        plt.xticks(x, benchmark_types)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, f"{pair}_{model_identifier}_benchmark_comparison.png"), dpi=300)
        plt.close()
    
    def plot_data_type_comparison(self, pair_evaluation: Dict[str, Dict[str, Dict[str, Any]]], 
                               symbol: str, pair: str) -> None:
        """
        Plot comparison of results across different data types (Raw, Enhanced, Selected)
        
        Args:
            pair_evaluation: Dictionary containing evaluation results for different data types
            symbol: Symbol for the currency pair (e.g., 'E', 'G', 'J')
            pair: Currency pair being visualized
        """
        # ตรวจสอบว่ามีข้อมูลทั้ง 3 ประเภทหรือไม่
        data_type_identifiers = [f"{symbol}1", f"{symbol}2", f"{symbol}3"]
        if not all(identifier in pair_evaluation for identifier in data_type_identifiers):
            print(f"Missing data types for {pair}. Skipping data type comparison.")
            return
        
        # Define data type labels
        data_type_labels = {
            f"{symbol}1": "Raw Data",
            f"{symbol}2": "Enhanced Features",
            f"{symbol}3": "Enhanced & Selected"
        }
        
        # กำหนดสี
        data_type_colors = {
            f"{symbol}1": 'lightblue',
            f"{symbol}2": 'skyblue',
            f"{symbol}3": 'steelblue'
        }
        
        # 1. แสดงกราฟเปรียบเทียบ RMSE
        plt.figure(figsize=(14, 8))
        
        models = list(pair_evaluation[f"{symbol}1"].keys())
        data_types = [1, 2, 3]
        
        x = np.arange(len(models))
        width = 0.8 / len(data_types)
        
        for i, data_type in enumerate(data_types):
            offset = (i - len(data_types) / 2 + 0.5) * width
            identifier = f"{symbol}{data_type}"
            values = [pair_evaluation[identifier][model]['metrics']['rmse'] for model in models]
            
            plt.bar(x + offset, values, width, label=data_type_labels[identifier], 
                  color=data_type_colors[identifier], alpha=0.9)
        
        plt.title(f'{pair} - RMSE Comparison Across Data Types')
        plt.ylabel('RMSE')
        plt.xlabel('Model')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, f"{pair}_data_type_rmse_comparison.png"), dpi=300)
        plt.close()
        
        # 2. แสดงกราฟเปรียบเทียบ Directional Accuracy
        plt.figure(figsize=(14, 8))
        
        for i, data_type in enumerate(data_types):
            offset = (i - len(data_types) / 2 + 0.5) * width
            identifier = f"{symbol}{data_type}"
            values = [pair_evaluation[identifier][model]['metrics']['directional_accuracy'] for model in models]
            
            plt.bar(x + offset, values, width, label=data_type_labels[identifier], 
                  color=data_type_colors[identifier], alpha=0.9)
        
        plt.title(f'{pair} - Directional Accuracy Comparison Across Data Types')
        plt.ylabel('Directional Accuracy (%)')
        plt.xlabel('Model')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, f"{pair}_data_type_directional_accuracy_comparison.png"), dpi=300)
        plt.close()
        
        # 3. แสดงกราฟเปรียบเทียบ Annual Return
        plt.figure(figsize=(14, 8))
        
        for i, data_type in enumerate(data_types):
            offset = (i - len(data_types) / 2 + 0.5) * width
            identifier = f"{symbol}{data_type}"
            values = [pair_evaluation[identifier][model]['trading_metrics']['annual_return'] for model in models]
            
            plt.bar(x + offset, values, width, label=data_type_labels[identifier], 
                  color=data_type_colors[identifier], alpha=0.9)
        
        plt.title(f'{pair} - Annual Return Comparison Across Data Types')
        plt.ylabel('Annual Return (%)')
        plt.xlabel('Model')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, f"{pair}_data_type_annual_return_comparison.png"), dpi=300)
        plt.close()
    
    def plot_bagging_comparison(self, evaluation_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Plot comparison of bagging approach vs. single pair approaches
        
        Args:
            evaluation_results: Dictionary containing evaluation results for all approaches
        """
        # ตรวจสอบว่ามีข้อมูลทั้ง E3, G3, J3 และ B หรือไม่
        pairs = self.config.CURRENCY_PAIRS
        symbols = {pair: self.config.PAIR_SYMBOLS[pair] for pair in pairs}
        
        # Define category labels
        category_labels = {
            f"{symbols['EURUSD']}3": "EURUSD (Enhanced & Selected)",
            f"{symbols['GBPUSD']}3": "GBPUSD (Enhanced & Selected)",
            f"{symbols['USDJPY']}3": "USDJPY (Enhanced & Selected)",
            'B': "Bagging Approach"
        }
        
        # Define category colors
        category_colors = {
            f"{symbols['EURUSD']}3": 'lightblue',
            f"{symbols['GBPUSD']}3": 'lightgreen',
            f"{symbols['USDJPY']}3": 'lightsalmon',
            'B': 'mediumpurple'
        }
        
        # 1. แสดงกราฟเปรียบเทียบ RMSE
        plt.figure(figsize=(14, 8))
        
        models = ['LSTM', 'GRU', 'XGBoost', 'TFT']
        categories = [f"{symbols[pair]}3" for pair in pairs] + ['B']
        
        x = np.arange(len(models))
        width = 0.8 / len(categories)
        
        # Collect RMSE values
        rmse_values = {}
        
        # Collect E3, G3, J3 values
        for pair in pairs:
            symbol = symbols[pair]
            identifier = f"{symbol}3"
            
            if pair in evaluation_results and identifier in evaluation_results[pair]:
                rmse_values[identifier] = [evaluation_results[pair][identifier][model]['metrics']['rmse'] for model in models]
        
        # Collect B values
        if 'bagging' in evaluation_results:
            rmse_values['B'] = []
            for model in models:
                # Calculate average across all pairs
                avg_rmse = np.mean([evaluation_results['bagging'][pair][model]['metrics']['rmse'] for pair in pairs])
                rmse_values['B'].append(avg_rmse)
        
        # Plot bars
        for i, category in enumerate(categories):
            if category in rmse_values:
                offset = (i - len(categories) / 2 + 0.5) * width
                plt.bar(x + offset, rmse_values[category], width, label=category_labels[category], 
                      color=category_colors[category], alpha=0.9)
        
        plt.title('RMSE Comparison: Single Pair vs Bagging Approach')
        plt.ylabel('RMSE')
        plt.xlabel('Model')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, "bagging_rmse_comparison.png"), dpi=300)
        plt.close()
        
        # 2. แสดงกราฟเปรียบเทียบ Directional Accuracy
        plt.figure(figsize=(14, 8))
        
        # Collect Directional Accuracy values
        dir_acc_values = {}
        
        # Collect E3, G3, J3 values
        for pair in pairs:
            symbol = symbols[pair]
            identifier = f"{symbol}3"
            
            if pair in evaluation_results and identifier in evaluation_results[pair]:
                dir_acc_values[identifier] = [evaluation_results[pair][identifier][model]['metrics']['directional_accuracy'] for model in models]
        
        # Collect B values
        if 'bagging' in evaluation_results:
            dir_acc_values['B'] = []
            for model in models:
                # Calculate average across all pairs
                avg_dir_acc = np.mean([evaluation_results['bagging'][pair][model]['metrics']['directional_accuracy'] for pair in pairs])
                dir_acc_values['B'].append(avg_dir_acc)
        
        # Plot bars
        for i, category in enumerate(categories):
            if category in dir_acc_values:
                offset = (i - len(categories) / 2 + 0.5) * width
                plt.bar(x + offset, dir_acc_values[category], width, label=category_labels[category], 
                      color=category_colors[category], alpha=0.9)
        
        plt.title('Directional Accuracy Comparison: Single Pair vs Bagging Approach')
        plt.ylabel('Directional Accuracy (%)')
        plt.xlabel('Model')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, "bagging_directional_accuracy_comparison.png"), dpi=300)
        plt.close()
        
        # 3. แสดงกราฟเปรียบเทียบ Annual Return
        plt.figure(figsize=(14, 8))
        
        # Collect Annual Return values
        return_values = {}
        
        # Collect E3, G3, J3 values
        for pair in pairs:
            symbol = symbols[pair]
            identifier = f"{symbol}3"
            
            if pair in evaluation_results and identifier in evaluation_results[pair]:
                return_values[identifier] = [evaluation_results[pair][identifier][model]['trading_metrics']['annual_return'] for model in models]
        
        # Collect B values
        if 'bagging' in evaluation_results:
            return_values['B'] = []
            for model in models:
                # Calculate average across all pairs
                avg_return = np.mean([evaluation_results['bagging'][pair][model]['trading_metrics']['annual_return'] for pair in pairs])
                return_values['B'].append(avg_return)
        
        # Plot bars
        for i, category in enumerate(categories):
            if category in return_values:
                offset = (i - len(categories) / 2 + 0.5) * width
                plt.bar(x + offset, return_values[category], width, label=category_labels[category], 
                      color=category_colors[category], alpha=0.9)
        
        plt.title('Annual Return Comparison: Single Pair vs Bagging Approach')
        plt.ylabel('Annual Return (%)')
        plt.xlabel('Model')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, "bagging_annual_return_comparison.png"), dpi=300)
        plt.close()
    
    def plot_feature_importance(self, importance_df: Any, pair: str, top_n: int = 20) -> None:
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame containing feature importance
            pair: Currency pair
            top_n: Number of top features to display
        """
        plt.figure(figsize=(12, 10))
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar plot
        bars = plt.barh(top_features['Feature'], top_features['Importance'], 
                       color='skyblue', alpha=0.8)
        
        # Add values to the bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.001, 
                   bar.get_y() + bar.get_height()/2, 
                   f"{top_features['Importance'].iloc[i]:.4f}", 
                   va='center', fontweight='bold')
        
        plt.title(f'Top {top_n} Features - {pair}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.config.RESULTS_PATH, f"{pair}_feature_importance.png"), dpi=300)
        plt.close()
    
    def plot_training_history(self, history: Any, model_name: str, pair: str) -> None:
        """
        Plot training history (loss and metrics over epochs)
        
        Args:
            history: Training history object from model training
            model_name: Name of the model
            pair: Currency pair
        """
        plt.figure(figsize=(14, 6))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.title(f'{model_name} - {pair} Loss During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot training & validation RMSE if available
        if 'root_mean_squared_error' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['root_mean_squared_error'], 
                   label='Training RMSE', color='blue')
            plt.plot(history.history['val_root_mean_squared_error'], 
                   label='Validation RMSE', color='red')
            plt.title(f'{model_name} - {pair} RMSE During Training')
            plt.xlabel('Epochs')
            plt.ylabel('RMSE')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, f"{pair}_{model_name}_training_history.png"), dpi=300)
        plt.close()
    
    def plot_balance_curve(self, trading_metrics: Dict[str, Any], model_name: str, pair: str) -> None:
        """
        Plot balance curve and drawdown from trading simulation
        
        Args:
            trading_metrics: Dictionary containing trading metrics
            model_name: Name of the model
            pair: Currency pair
        """
        plt.figure(figsize=(14, 10))
        
        # Plot balance curve
        plt.subplot(2, 1, 1)
        plt.plot(trading_metrics['balance_curve'], label='Account Balance', color='green')
        plt.title(f'{model_name} - {pair} Account Balance Curve')
        plt.xlabel('Trades')
        plt.ylabel('Balance')
        plt.axhline(y=self.config.INITIAL_BALANCE, color='red', linestyle='--', 
                  label=f'Initial Balance ({self.config.INITIAL_BALANCE})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Annotate final balance
        final_balance = trading_metrics['balance_curve'][-1]
        plt.annotate(f'Final Balance: {final_balance:.2f}', 
                   xy=(len(trading_metrics['balance_curve'])-1, final_balance),
                   xytext=(0.8, 0.9), textcoords='axes fraction',
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                   fontsize=12, fontweight='bold')
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        plt.fill_between(range(len(trading_metrics['drawdown_curve'])), 
                        0, trading_metrics['drawdown_curve'], 
                        color='red', alpha=0.3)
        plt.plot(trading_metrics['drawdown_curve'], color='red', label='Drawdown')
        plt.title(f'{model_name} - {pair} Drawdown')
        plt.xlabel('Trades')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Annotate maximum drawdown
        max_dd = np.max(trading_metrics['drawdown_curve'])
        max_dd_idx = np.argmax(trading_metrics['drawdown_curve'])
        plt.annotate(f'Max Drawdown: {max_dd:.2f}%', 
                   xy=(max_dd_idx, max_dd),
                   xytext=(0.8, 0.9), textcoords='axes fraction',
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                   fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_PATH, f"{pair}_{model_name}_balance_curve.png"), dpi=300)
        plt.close()