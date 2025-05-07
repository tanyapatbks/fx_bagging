"""
Hyperparameter Tuning Module for Forex Prediction
This module implements various hyperparameter tuning techniques.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Callable, Optional
import time
import json
from sklearn.model_selection import KFold, train_test_split
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt

# Import project modules
from config.config import Config
from src.stage3_prediction_models.lstm_model import LSTMModel
from src.stage3_prediction_models.gru_model import GRUModel
from src.stage3_prediction_models.xgboost_model import XGBoostModel
from src.stage3_prediction_models.tft_model import TFTModel


class HyperparameterTuner:
    """
    Class for tuning hyperparameters of various models
    """
    
    def __init__(self, config, data_container, model_type, pair, data_type='selected'):
        """
        Initialize the HyperparameterTuner
        
        Args:
            config: Configuration object
            data_container: Container with training and validation data
            model_type: Type of model ('LSTM', 'GRU', 'XGBoost', 'TFT')
            pair: Currency pair being tuned
            data_type: Type of data ('raw', 'enhanced', 'selected')
        """
        self.config = config
        self.data_container = data_container
        self.model_type = model_type
        self.pair = pair
        self.data_type = data_type
        self.results_path = os.path.join(config.HYPERPARAMS_PATH, pair)
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
            
        # Extract the necessary data based on model type
        if model_type == 'XGBoost':
            # For XGBoost, use tabular data
            self.X_train = data_container['X_tab_train']
            self.y_train = data_container['y_tab_train']
            self.scaler = data_container['tab_scaler']
            self.target_idx = data_container['tab_target_idx']
        else:
            # For sequence models (LSTM, GRU, TFT)
            self.X_train = data_container['X_seq_train']
            self.y_train = data_container['y_seq_train']
            self.scaler = data_container['seq_scaler']
            self.target_idx = data_container['target_idx']
        
        # Split the training data into train and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, shuffle=False
        )
        
        print(f"Prepared data for tuning {model_type} on {pair} using {data_type} data")
        print(f"Training set: {self.X_train.shape}, Validation set: {self.X_val.shape}")
    
    def _create_model(self, params: Dict[str, Any]) -> Any:
        """
        Create model with given hyperparameters
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            Model instance
        """
        if self.model_type == 'LSTM':
            return LSTMModel(self.config, params)
        elif self.model_type == 'GRU':
            return GRUModel(self.config, params)
        elif self.model_type == 'XGBoost':
            return XGBoostModel(self.config, params)
        elif self.model_type == 'TFT':
            return TFTModel(self.config, params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_lstm_search_space(self, trial):
        """Define search space for LSTM hyperparameters"""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'units_layer1': trial.suggest_int('units_layer1', 50, 256),
            'units_layer2': trial.suggest_int('units_layer2', 25, 128),
            'dropout1': trial.suggest_float('dropout1', 0.1, 0.5),
            'dropout2': trial.suggest_float('dropout2', 0.1, 0.5),
            'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.0, 0.3),
            'l1_reg': trial.suggest_float('l1_reg', 1e-5, 1e-3, log=True),
            'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-3, log=True)
        }
    
    def _get_gru_search_space(self, trial):
        """Define search space for GRU hyperparameters"""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'units_layer1': trial.suggest_int('units_layer1', 50, 256),
            'units_layer2': trial.suggest_int('units_layer2', 25, 128),
            'dropout1': trial.suggest_float('dropout1', 0.1, 0.5),
            'dropout2': trial.suggest_float('dropout2', 0.1, 0.5),
            'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.0, 0.3),
            'l1_reg': trial.suggest_float('l1_reg', 1e-5, 1e-3, log=True),
            'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-3, log=True)
        }
    
    def _get_xgboost_search_space(self, trial):
        """Define search space for XGBoost hyperparameters"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 2.0),
            'random_state': 42
        }
    
    def _get_tft_search_space(self, trial):
        """Define search space for TFT hyperparameters"""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'attention_heads': trial.suggest_int('attention_heads', 1, 8),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'hidden_units': trial.suggest_int('hidden_units', 32, 128),
            'hidden_continuous_size': trial.suggest_int('hidden_continuous_size', 16, 64)
        }
    
    def _objective_function(self, trial) -> float:
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation metric (to be minimized)
        """
        # Define hyperparameter search space based on model type
        if self.model_type == 'LSTM':
            params = self._get_lstm_search_space(trial)
        elif self.model_type == 'GRU':
            params = self._get_gru_search_space(trial)
        elif self.model_type == 'XGBoost':
            params = self._get_xgboost_search_space(trial)
        elif self.model_type == 'TFT':
            params = self._get_tft_search_space(trial)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        try:
            # Create model with trial hyperparameters
            model_instance = self._create_model(params)
            
            # Handle different model types for training and evaluation
            if self.model_type == 'XGBoost':
                # For XGBoost, we handle differently since it's not a neural network
                model = model_instance.train(self.X_train, self.y_train, 
                                          eval_set=[(self.X_val, self.y_val)])
                
                if model is None:
                    return float('inf')  # Return a large value if model training fails
                
                y_pred = model_instance.predict(model, self.X_val)
                
                # Calculate RMSE as the objective
                rmse = np.sqrt(np.mean((self.y_val - y_pred) ** 2))
                
                # Calculate directional accuracy for reporting
                # Calculate price direction (up/down)
                true_direction = np.diff(self.y_val) > 0
                pred_direction = np.diff(y_pred) > 0
                
                # Calculate direction accuracy
                direction_matches = true_direction == pred_direction
                directional_accuracy = np.mean(direction_matches) * 100
                
                # Report additional metrics
                trial.set_user_attr('directional_accuracy', directional_accuracy)
                
                # Calculate objective - we want to maximize directional accuracy and minimize RMSE
                # So we create a custom objective that considers both
                # This weights directional accuracy more heavily than RMSE
                # We will return a negative value so Optuna minimizes it (which maximizes our objective)
                objective = rmse * (100 - directional_accuracy) / 100
                
                return objective
            else:
                # For neural network models
                # Use a custom callback to record best metrics
                best_metrics = {'val_loss': float('inf'), 'val_rmse': float('inf'), 'directional_accuracy': 0}
                
                class MetricsCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        if logs['val_loss'] < best_metrics['val_loss']:
                            best_metrics['val_loss'] = logs['val_loss']
                            best_metrics['val_rmse'] = logs['val_root_mean_squared_error']
                            
                            # Calculate directional accuracy on validation set
                            y_pred = self.model.predict(self.validation_data[0])
                            
                            # Reshape predictions if needed
                            if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                                y_pred = y_pred.flatten()
                            
                            true_val = self.validation_data[1]
                            
                            # Calculate direction
                            true_direction = np.diff(true_val) > 0
                            pred_direction = np.diff(y_pred) > 0
                            
                            # Calculate accuracy
                            direction_matches = true_direction == pred_direction
                            best_metrics['directional_accuracy'] = np.mean(direction_matches) * 100
                
                # Create callback with validation data
                metrics_callback = MetricsCallback()
                metrics_callback.validation_data = (self.X_val, self.y_val)
                
                # Create standard callbacks
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', 
                        patience=self.config.PATIENCE // 2,  # Reduce patience for faster tuning
                        restore_best_weights=True,
                        verbose=0
                    ),
                    metrics_callback
                ]
                
                # Train model with reduced epochs for faster tuning
                model, history = model_instance.train(
                    self.X_train, self.y_train, 
                    f"{self.pair}_{self.model_type}_trial_{trial.number}",
                    callbacks=callbacks,
                    epochs=50  # Reduced epochs for faster tuning
                )
                
                # Get validation loss from history
                val_loss = best_metrics['val_loss']
                directional_accuracy = best_metrics['directional_accuracy']
                
                # Report additional metrics
                trial.set_user_attr('directional_accuracy', directional_accuracy)
                trial.set_user_attr('val_rmse', best_metrics['val_rmse'])
                
                # Use custom objective that balances RMSE and directional accuracy
                objective = val_loss * (100 - directional_accuracy) / 100
                
                return objective
                
        except Exception as e:
            print(f"Error in trial {trial.number}: {e}")
            return float('inf')  # Return a large value if an error occurs
    
    def run_optuna_optimization(self, n_trials=30, timeout=7200):
        """
        Run Optuna optimization to find the best hyperparameters
        
        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            
        Returns:
            Best hyperparameters and study object
        """
        # Create study with TPE sampler and median pruner
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Run optimization
        print(f"\nStarting hyperparameter optimization for {self.model_type} on {self.pair}")
        print(f"Running {n_trials} trials with a timeout of {timeout} seconds...")
        
        start_time = time.time()
        study.optimize(self._objective_function, n_trials=n_trials, timeout=timeout)
        end_time = time.time()
        
        # Get best trial and parameters
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        print(f"\nOptimization completed in {(end_time - start_time) / 60:.2f} minutes.")
        print(f"Best trial: {best_trial.number}")
        print(f"Best value: {best_value}")
        print(f"Best hyperparameters: {best_params}")
        
        if 'directional_accuracy' in best_trial.user_attrs:
            print(f"Directional Accuracy: {best_trial.user_attrs['directional_accuracy']:.2f}%")
        
        # Save results
        results = {
            'model_type': self.model_type,
            'pair': self.pair,
            'data_type': self.data_type,
            'best_trial': best_trial.number,
            'best_value': best_value,
            'best_params': best_params,
            'additional_metrics': {k: v for k, v in best_trial.user_attrs.items()},
            'optimization_time': end_time - start_time,
            'n_trials': n_trials,
            'trials_history': [{
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'metrics': {k: v for k, v in t.user_attrs.items() if k in ['directional_accuracy', 'val_rmse']}
            } for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        }
        
        results_file = os.path.join(self.results_path, f"{self.model_type}_{self.data_type}_hyperparams.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {results_file}")
        
        # Generate and save visualization
        self.plot_optimization_history(study)
        
        # Return best parameters and study
        return best_params, study
    
    def plot_optimization_history(self, study):
        """
        Plot optimization history and parameter importance
        
        Args:
            study: Optuna study object
        """
        # Create plots directory
        plots_dir = os.path.join(self.results_path, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Plot optimization history
        plt.figure(figsize=(12, 6))
        
        # Extract data
        trials = study.trials
        values = [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        best_values = np.minimum.accumulate(values)
        
        # Plot optimization history
        plt.subplot(1, 2, 1)
        plt.plot(values, 'o-', alpha=0.7, label='Trial Value')
        plt.plot(best_values, 'r-', linewidth=2, label='Best Value')
        plt.xlabel('Trial Number')
        plt.ylabel('Objective Value')
        plt.title('Optimization History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot parameter importance
        plt.subplot(1, 2, 2)
        
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            importances = list(importance.values())
            
            # Sort by importance
            sorted_indices = np.argsort(importances)[::-1]
            sorted_params = [params[i] for i in sorted_indices]
            sorted_importances = [importances[i] for i in sorted_indices]
            
            # Plot top 10 parameters
            top_n = min(10, len(sorted_params))
            plt.barh(range(top_n), sorted_importances[:top_n], align='center')
            plt.yticks(range(top_n), sorted_params[:top_n])
            plt.xlabel('Importance')
            plt.title('Parameter Importance')
            plt.grid(True, alpha=0.3)
        except Exception as e:
            plt.text(0.5, 0.5, f"Could not compute parameter importance: {e}", 
                   ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{self.model_type}_{self.data_type}_optimization.png"), dpi=300)
        plt.close()
        
        # Also plot individual parameter effects if possible
        try:
            # Choose top parameters by importance
            top_params = sorted_params[:min(6, len(sorted_params))]
            
            # Create plot
            n_params = len(top_params)
            n_cols = 3
            n_rows = (n_params + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 4 * n_rows))
            
            # Plot each parameter
            for i, param in enumerate(top_params):
                plt.subplot(n_rows, n_cols, i + 1)
                
                # Get parameter values and corresponding trial values
                param_values = [t.params[param] if param in t.params else None 
                               for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
                trial_values = values
                
                # Remove None values
                valid_indices = [i for i, v in enumerate(param_values) if v is not None]
                valid_param_values = [param_values[i] for i in valid_indices]
                valid_trial_values = [trial_values[i] for i in valid_indices]
                
                # Sort by parameter value
                sorted_indices = np.argsort(valid_param_values)
                sorted_param_values = [valid_param_values[i] for i in sorted_indices]
                sorted_trial_values = [valid_trial_values[i] for i in sorted_indices]
                
                # Plot
                plt.scatter(sorted_param_values, sorted_trial_values, alpha=0.7)
                plt.xlabel(param)
                plt.ylabel('Objective Value')
                plt.title(f'Effect of {param}')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{self.model_type}_{self.data_type}_param_effects.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Could not plot parameter effects: {e}")