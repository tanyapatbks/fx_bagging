"""
Main Entry Point for the Forex Prediction System

This script orchestrates the entire workflow of the Forex Prediction System:
1. Data Acquisition
2. Feature Engineering
3. Model Training and Prediction
4. Evaluation and Visualization

Command-line arguments:
    --mode: Mode to run (train, visualize, eval, or all)
    --pair: Currency pair to process (EURUSD, GBPUSD, USDJPY, or all)
    --model: Model type to use (LSTM, GRU, XGBoost, TFT, or all)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Import project modules
from config.config import Config
from src.stage1_data_acquisition import DataLoader
from src.stage2_feature_engineering import FeatureEngineer
from src.stage3_prediction_models import LSTMModel, GRUModel, XGBoostModel, TFTModel
from src.stage4_evaluation import ModelEvaluator
from src.visualization import ResultVisualizer
from src.reporting import ReportGenerator
from utils.data_utils import SequenceDataHandler


class ForexPredictionSystem:
    """
    Main class that orchestrates the entire workflow
    """
    def __init__(self, config):
        """
        Initialize the ForexPredictionSystem with configuration
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.config.create_directories()
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.feature_engineer = FeatureEngineer(config)
        self.seq_data_handler = SequenceDataHandler(config)
        self.evaluator = ModelEvaluator(config)
        self.visualizer = ResultVisualizer(config)
        self.reporter = ReportGenerator(config)
        
        # Initialize models
        self.lstm_model = LSTMModel(config)
        self.gru_model = GRUModel(config)
        self.xgb_model = XGBoostModel(config)
        self.tft_model = TFTModel(config)
        
        # Storage for results
        self.pair_data = {}
        self.processed_data = {}
        self.model_data = {}
        self.results = {}
        self.evaluation_results = {}
    
    def run_data_acquisition(self):
        """
        Stage 1: Load and preprocess data for all currency pairs
        """
        print("\n" + "="*60)
        print("Stage 1: Data Acquisition")
        print("="*60)
        
        # Load data for all pairs
        self.pair_data = self.data_loader.load_all_pairs()
        
        # Check if data was loaded successfully
        for pair in self.config.CURRENCY_PAIRS:
            if pair not in self.pair_data:
                print(f"ERROR: Failed to load data for {pair}")
                return False
            
            print(f"Successfully loaded {pair} data")
            print(f"  Training data: {self.pair_data[pair]['train'].shape}")
            print(f"  Testing data: {self.pair_data[pair]['test'].shape}")
        
        return True
    
    def run_feature_engineering(self):
        """
        Stage 2: Enhance features and perform feature selection
        """
        print("\n" + "="*60)
        print("Stage 2: Feature Engineering")
        print("="*60)
        
        self.processed_data = {}
        
        for pair in self.config.CURRENCY_PAIRS:
            print(f"\n--- Processing {pair} ---")
            
            # Get raw data
            train_df = self.pair_data[pair]['train']
            test_df = self.pair_data[pair]['test']
            
            # 1. Raw Data - ไม่มีการเพิ่ม features
            print("1. Preparing Raw Data...")
            
            # 2. Enhanced Data - เพิ่ม Technical Indicators
            print("2. Adding Technical Indicators...")
            train_df_enhanced = self.feature_engineer.add_technical_indicators(train_df)
            test_df_enhanced = self.feature_engineer.add_technical_indicators(test_df)
            
            # 3. Enhanced + Selected Data - เพิ่ม Technical Indicators แล้วทำ Feature Selection
            print("3. Selecting Important Features...")
            train_df_selected, test_df_selected, importance = self.feature_engineer.select_features(
                train_df_enhanced, test_df_enhanced
            )
            
            # Save feature importance visualization
            self.visualizer.plot_feature_importance(importance, pair)
            
            # เก็บข้อมูลทั้ง 3 แบบไว้
            self.processed_data[pair] = {
                'raw': {
                    'train': train_df,
                    'test': test_df
                },
                'enhanced': {
                    'train': train_df_enhanced,
                    'test': test_df_enhanced
                },
                'selected': {
                    'train': train_df_selected,
                    'test': test_df_selected
                },
                'feature_importance': importance
            }
        
        return True
    
    def prepare_model_data(self):
        """
        Prepare data for model training
        """
        print("\n" + "="*60)
        print("Preparing Model Data")
        print("="*60)
        
        model_data = {}
        
        for pair in self.config.CURRENCY_PAIRS:
            print(f"\n--- Preparing Model Data for {pair} ---")
            pair_data = self.processed_data[pair]
            model_data[pair] = {}
            
            # เตรียมข้อมูลสำหรับทั้ง 3 ประเภท
            for data_type in ['raw', 'enhanced', 'selected']:
                train_df = pair_data[data_type]['train']
                test_df = pair_data[data_type]['test']
                
                print(f"Preparing {data_type} data...")
                
                # เตรียมข้อมูลสำหรับโมเดล sequence (LSTM, GRU, TFT)
                X_seq_train, y_seq_train, seq_scaler, target_idx, _ = self.seq_data_handler.prepare_sequence_data(
                    train_df
                )
                X_seq_test, y_seq_test, _, _, _ = self.seq_data_handler.prepare_sequence_data(
                    test_df
                )
                
                # เตรียมข้อมูลสำหรับโมเดล tabular (XGBoost)
                X_tab_train, y_tab_train, tab_scaler, tab_target_idx, feature_cols = self.seq_data_handler.prepare_tabular_data(
                    train_df
                )
                X_tab_test, y_tab_test, _, _, _ = self.seq_data_handler.prepare_tabular_data(
                    test_df
                )
                
                # เก็บข้อมูล
                model_data[pair][data_type] = {
                    'X_seq_train': X_seq_train,
                    'y_seq_train': y_seq_train,
                    'X_seq_test': X_seq_test,
                    'y_seq_test': y_seq_test,
                    'seq_scaler': seq_scaler,
                    'target_idx': target_idx,
                    'X_tab_train': X_tab_train,
                    'y_tab_train': y_tab_train,
                    'X_tab_test': X_tab_test,
                    'y_tab_test': y_tab_test,
                    'tab_scaler': tab_scaler,
                    'tab_target_idx': tab_target_idx,
                    'feature_cols': feature_cols
                }
        
        # เตรียมข้อมูลสำหรับ Bagging Approach
        print("\n--- Preparing Bagging Approach Data ---")
        
        # ในกรณีนี้เราจะใช้ข้อมูล 'selected' จากทั้ง 3 คู่เงิน
        bagging_data = self.seq_data_handler.prepare_bagging_data(model_data)
        model_data['bagging'] = bagging_data
        
        self.model_data = model_data
        return True
    
    def train_single_model(self, model_type, X_train, y_train, model_name, X_test, scaler, target_idx):
        """
        Train a single model and make predictions
        
        Args:
            model_type: Type of model to train (LSTM, GRU, XGBoost, TFT)
            X_train: Training features
            y_train: Training targets
            model_name: Name for the saved model
            X_test: Test features
            scaler: Scaler used for normalization
            target_idx: Index of the target column
            
        Returns:
            Tuple of (model, history, predictions)
        """
        if model_type == 'LSTM':
            model, history = self.lstm_model.train(X_train, y_train, model_name)
            predictions = self.lstm_model.predict(model, X_test, scaler, target_idx)
            return model, history, predictions
        
        elif model_type == 'GRU':
            model, history = self.gru_model.train(X_train, y_train, model_name)
            predictions = self.gru_model.predict(model, X_test, scaler, target_idx)
            return model, history, predictions
        
        elif model_type == 'XGBoost':
            # For XGBoost, we need tabular data and different parameters
            # Check if we have a tuple of (X_train, y_train, X_test, scaler, target_idx)
            if isinstance(X_train, tuple) and len(X_train) == 5:
                X_train_tab, y_train_tab, X_test_tab, tab_scaler, tab_target_idx = X_train
                
                # Split data for early stopping
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train_tab, y_train_tab, test_size=self.config.VALIDATION_SPLIT, shuffle=False
                )
                
                model = self.xgb_model.train(X_tr, y_tr, eval_set=[(X_val, y_val)])
                if model is not None:
                    self.xgb_model.save_model(model, model_name)
                
                predictions = self.xgb_model.predict(model, X_test_tab)
                # No history for XGBoost
                return model, None, predictions
            else:
                print(f"Error: XGBoost requires tabular data, but received sequence data")
                return None, None, None
        
        elif model_type == 'TFT':
            model, history = self.tft_model.train(X_train, y_train, model_name)
            predictions = self.tft_model.predict(model, X_test, scaler, target_idx)
            return model, history, predictions
        
        else:
            print(f"Error: Unknown model type {model_type}")
            return None, None, None
    
    def run_model_training(self, model_types=None, pairs=None):
        """
        Stage 3: Train prediction models
        
        Args:
            model_types: List of model types to train (default: all)
            pairs: List of currency pairs to process (default: all)
        """
        print("\n" + "="*60)
        print("Stage 3: Model Training")
        print("="*60)
        
        # Default is to train all models for all pairs
        if model_types is None:
            model_types = ['LSTM', 'GRU', 'XGBoost', 'TFT']
        
        if pairs is None:
            pairs = self.config.CURRENCY_PAIRS + ['bagging']
        
        # Results storage
        results = {}
        
        # 1. Train models for individual pairs
        for pair in [p for p in pairs if p != 'bagging']:
            symbol = self.config.PAIR_SYMBOLS[pair]
            print(f"\n--- Training Models for {pair} [{symbol}] ---")
            pair_results = {}
            
            # Train models for each data type (raw, enhanced, selected)
            for idx, data_type in enumerate(['raw', 'enhanced', 'selected'], 1):
                data = self.model_data[pair][data_type]
                model_identifier = f"{symbol}{idx}"  # E1, E2, E3, G1, G2, G3, J1, J2, J3
                
                print(f"\nTraining {data_type} models [{model_identifier}]...")
                
                # Storage for models, histories, and predictions
                model_results = {}
                
                # Train each model type
                for model_type in model_types:
                    print(f"Training {model_type} model for {pair} {data_type}...")
                    
                    if model_type == 'XGBoost':
                        # For XGBoost, use tabular data
                        X_train_tuple = (
                            data['X_tab_train'], 
                            data['y_tab_train'],
                            data['X_tab_test'],
                            data['tab_scaler'],
                            data['tab_target_idx']
                        )
                        
                        model, history, predictions = self.train_single_model(
                            model_type,
                            X_train_tuple,  # Pass the tuple of data
                            None,  # Not used for XGBoost
                            f"{pair}_{data_type}_{model_type}",
                            None,  # Not used for XGBoost
                            None,  # Not used for XGBoost
                            None   # Not used for XGBoost
                        )
                    else:
                        # For sequence models (LSTM, GRU, TFT)
                        model, history, predictions = self.train_single_model(
                            model_type,
                            data['X_seq_train'],
                            data['y_seq_train'],
                            f"{pair}_{data_type}_{model_type}",
                            data['X_seq_test'],
                            data['seq_scaler'],
                            data['target_idx']
                        )
                    
                    # Store results
                    actual_values = data['y_seq_test']
                    actual_dates = self.processed_data[pair][data_type]['test']['Time'].values[self.config.SEQUENCE_LENGTH:]
                    
                    model_results[model_type] = {
                        'model': model,
                        'history': history,
                        'predictions': predictions,
                        'actual': actual_values,
                        'dates': actual_dates
                    }
                    
                    # Plot training history if available
                    if history is not None:
                        self.visualizer.plot_training_history(history, f"{data_type}_{model_type}", pair)
                
                # Store results for this data type
                pair_results[model_identifier] = model_results
            
            # Store results for this pair
            results[pair] = pair_results
        
        # 2. Train models for bagging approach
        if 'bagging' in pairs:
            print("\n--- Training Bagging Approach [B] ---")
            
            # Use data from all pairs (E3, G3, J3)
            bagging_data = self.model_data['bagging']
            bagging_results = {}
            
            # Train each model type
            for model_type in model_types:
                print(f"Training {model_type} model for Bagging...")
                
                if model_type == 'XGBoost':
                    # For XGBoost, use tabular data
                    # Split data for early stopping
                    X_train, X_val, y_train, y_val = train_test_split(
                        bagging_data['X_tab_train'], 
                        bagging_data['y_tab_train'],
                        test_size=self.config.VALIDATION_SPLIT,
                        shuffle=False
                    )
                    
                    bagging_model = self.xgb_model.train(
                        X_train, y_train, eval_set=[(X_val, y_val)]
                    )
                    
                    if bagging_model is not None:
                        self.xgb_model.save_model(bagging_model, f"Bagging_{model_type}")
                    
                    history = None
                else:
                    # For sequence models (LSTM, GRU, TFT)
                    bagging_model, history = getattr(self, f"{model_type.lower()}_model").train(
                        bagging_data['X_seq_train'], 
                        bagging_data['y_seq_train'], 
                        f"Bagging_{model_type}"
                    )
                
                # Store model and history
                for i, pair in enumerate(self.config.CURRENCY_PAIRS):
                    print(f"Predicting {pair} with Bagging {model_type} model...")
                    
                    if model_type == 'XGBoost':
                        # For XGBoost, use tabular data
                        predictions = self.xgb_model.predict(
                            bagging_model, 
                            bagging_data['X_tab_test_pairs'][i]
                        )
                    else:
                        # For sequence models
                        predictions = getattr(self, f"{model_type.lower()}_model").predict(
                            bagging_model, 
                            bagging_data['X_seq_test_pairs'][i], 
                            bagging_data['seq_scalers'][pair], 
                            bagging_data['target_idxs'][pair]
                        )
                    
                    # Get actual values
                    actual_values = bagging_data['y_seq_test_pairs'][i]
                    actual_dates = self.processed_data[pair]['selected']['test']['Time'].values[self.config.SEQUENCE_LENGTH:]
                    
                    if pair not in bagging_results:
                        bagging_results[pair] = {}
                    
                    bagging_results[pair][model_type] = {
                        'model': bagging_model,
                        'history': history,
                        'predictions': predictions,
                        'actual': actual_values,
                        'dates': actual_dates
                    }
                
                # Plot training history if available
                if history is not None:
                    self.visualizer.plot_training_history(history, f"Bagging_{model_type}", "Combined")
            
            # Store bagging results
            results['bagging'] = bagging_results
        
        self.results = results
        return True
    
    def run_hyperparameter_tuning(self, model_types=None, pairs=None, data_types=None, 
                             n_trials=30, save_best=True):
        """
        Run hyperparameter tuning for specified models and pairs
        
        Args:
            model_types: List of model types to tune (default: all)
            pairs: List of currency pairs to process (default: all)
            data_types: List of data types to use (default: 'selected')
            n_trials: Number of trials per tuning run
            save_best: Whether to save best parameters to config
        """
        print("\n" + "="*60)
        print("Hyperparameter Tuning")
        print("="*60)
        
        # Default is to tune all models for all pairs
        if model_types is None:
            model_types = ['LSTM', 'GRU', 'XGBoost', 'TFT']
        
        if pairs is None:
            pairs = self.config.CURRENCY_PAIRS
        
        if data_types is None:
            data_types = ['selected']  # Default to using feature-selected data
        
        # Need to have prepared model data
        if not hasattr(self, 'model_data') or not self.model_data:
            print("Model data not prepared. Running data preparation steps...")
            self.run_data_acquisition()
            self.run_feature_engineering()
            self.prepare_model_data()
        
        # Import the hyperparameter tuning module
        from src.hyperparameter_tuning import HyperparameterTuner
        
        # Dictionary to store best parameters for all models
        best_params_all = {}
        
        # For each pair and model type, run hyperparameter tuning
        for pair in pairs:
            best_params_all[pair] = {}
            
            for data_type in data_types:
                print(f"\n--- Tuning hyperparameters for {pair} using {data_type} data ---")
                
                # Get data for this pair and data type
                if pair not in self.model_data or data_type not in self.model_data[pair]:
                    print(f"Data not available for {pair} {data_type}. Skipping.")
                    continue
                
                data = self.model_data[pair][data_type]
                
                # Tuning each model type
                for model_type in model_types:
                    print(f"\nTuning {model_type} model for {pair} using {data_type} data...")
                    
                    # Create tuner
                    tuner = HyperparameterTuner(
                        self.config, 
                        data,
                        model_type, 
                        pair,
                        data_type
                    )
                    
                    # Run optimization
                    best_params, study = tuner.run_optuna_optimization(n_trials=n_trials)
                    
                    # Store best parameters
                    if model_type not in best_params_all[pair]:
                        best_params_all[pair][model_type] = {}
                    
                    best_params_all[pair][model_type][data_type] = best_params
        
        if save_best:
            # Find the best parameters across all pairs for each model
            for model_type in model_types:
                best_score = float('inf')
                best_model_params = None
                best_pair = None
                best_data_type = None
                
                # Find the best parameters for this model type
                for pair in pairs:
                    if pair in best_params_all and model_type in best_params_all[pair]:
                        for data_type in best_params_all[pair][model_type]:
                            # Load the results file to get the score
                            results_file = os.path.join(
                                self.config.HYPERPARAMS_PATH, 
                                pair, 
                                f"{model_type}_{data_type}_hyperparams.json"
                            )
                            
                            if os.path.exists(results_file):
                                with open(results_file, 'r') as f:
                                    results = json.load(f)
                                
                                # Get the score (directional accuracy is better)
                                if 'additional_metrics' in results and 'directional_accuracy' in results['additional_metrics']:
                                    # For directional accuracy, higher is better, so we use negative for comparison
                                    score = -results['additional_metrics']['directional_accuracy']
                                else:
                                    score = results['best_value']
                                
                                if score < best_score:
                                    best_score = score
                                    best_model_params = best_params_all[pair][model_type][data_type]
                                    best_pair = pair
                                    best_data_type = data_type
                
                # Update config with best parameters
                if best_model_params is not None:
                    if model_type == 'LSTM':
                        self.config.LSTM_PARAMS = best_model_params
                    elif model_type == 'GRU':
                        self.config.GRU_PARAMS = best_model_params
                    elif model_type == 'XGBoost':
                        self.config.XGB_PARAMS = best_model_params
                    elif model_type == 'TFT':
                        self.config.TFT_PARAMS = best_model_params
                    
                    print(f"\nUpdated {model_type} parameters with best from {best_pair} {best_data_type}")
                    print(f"Best parameters: {best_model_params}")
        
        # Save all best parameters to file
        best_params_file = os.path.join(self.config.HYPERPARAMS_PATH, "best_params_all.json")
        with open(best_params_file, 'w') as f:
            json.dump(best_params_all, f, indent=4)
        
        print("\nHyperparameter tuning completed.")
        print(f"All best parameters saved to {best_params_file}")
        if save_best:
            print("The best parameters have been updated in the config.")
        print("You can now run model training with the optimized parameters.")

    def run_evaluation(self):
        """
        Stage 4: Evaluate model predictions
        """
        print("\n" + "="*60)
        print("Stage 4: Evaluation")
        print("="*60)
        
        evaluation_results = {}
        
        # 1. Evaluate models for individual pairs
        for pair in self.config.CURRENCY_PAIRS:
            symbol = self.config.PAIR_SYMBOLS[pair]
            print(f"\n--- Evaluating Models for {pair} [{symbol}] ---")
            
            pair_evaluation = {}
            
            # Evaluate each data type (E1, E2, E3, G1, G2, G3, J1, J2, J3)
            for idx, data_type in enumerate(['raw', 'enhanced', 'selected'], 1):
                model_identifier = f"{symbol}{idx}"
                print(f"\nEvaluating {data_type} models [{model_identifier}]...")
                
                # Evaluate each model (LSTM, GRU, XGBoost, TFT)
                model_evaluation = {}
                for model_name in ['LSTM', 'GRU', 'XGBoost', 'TFT']:
                    if model_name not in self.results[pair][model_identifier]:
                        continue
                        
                    print(f"Evaluating {model_name}...")
                    
                    model_results = self.results[pair][model_identifier][model_name]
                    actual = model_results['actual']
                    predicted = model_results['predictions']
                    
                    # คำนวณเมตริกต่างๆ
                    metrics = self.evaluator.calculate_metrics(actual, predicted)
                    trading_metrics = self.evaluator.calculate_trading_metrics(actual, predicted)
                    benchmark = self.evaluator.compare_with_benchmarks(actual, predicted)
                    
                    # Plot balance curve
                    self.visualizer.plot_balance_curve(
                        trading_metrics, 
                        f"{model_identifier}_{model_name}", 
                        pair
                    )
                    
                    # บันทึกผลการประเมิน
                    model_evaluation[model_name] = {
                        'metrics': metrics,
                        'trading_metrics': trading_metrics,
                        'benchmark': benchmark
                    }
                
                # บันทึกผลการประเมินสำหรับประเภทข้อมูลนี้
                pair_evaluation[model_identifier] = model_evaluation
            
            # บันทึกผลการประเมินสำหรับคู่เงินนี้
            evaluation_results[pair] = pair_evaluation
        
        # 2. Evaluate Bagging Approach
        if 'bagging' in self.results:
            print("\n--- Evaluating Bagging Approach [B] ---")
            
            bagging_evaluation = {}
            for pair in self.config.CURRENCY_PAIRS:
                if pair not in self.results['bagging']:
                    continue
                    
                print(f"Evaluating Bagging results for {pair}...")
                
                # Evaluate each model (LSTM, GRU, XGBoost, TFT)
                model_evaluation = {}
                for model_name in ['LSTM', 'GRU', 'XGBoost', 'TFT']:
                    if model_name not in self.results['bagging'][pair]:
                        continue
                        
                    print(f"Evaluating {model_name}...")
                    
                    model_results = self.results['bagging'][pair][model_name]
                    actual = model_results['actual']
                    predicted = model_results['predictions']
                    
                    # คำนวณเมตริกต่างๆ
                    metrics = self.evaluator.calculate_metrics(actual, predicted)
                    trading_metrics = self.evaluator.calculate_trading_metrics(actual, predicted)
                    benchmark = self.evaluator.compare_with_benchmarks(actual, predicted)
                    
                    # Plot balance curve
                    self.visualizer.plot_balance_curve(
                        trading_metrics, 
                        f"Bagging_{model_name}", 
                        pair
                    )
                    
                    # บันทึกผลการประเมิน
                    model_evaluation[model_name] = {
                        'metrics': metrics,
                        'trading_metrics': trading_metrics,
                        'benchmark': benchmark
                    }
                
                # บันทึกผลการประเมินสำหรับ Bagging ของคู่เงินนี้
                bagging_evaluation[pair] = model_evaluation
            
            # บันทึกผลการประเมินสำหรับ Bagging
            evaluation_results['bagging'] = bagging_evaluation
        
        self.evaluation_results = evaluation_results
        return True
    
    def run_tuned_model_evaluation(self, model_types=None, pairs=None, data_types=None):
        """
        Train and evaluate models with hyperparameters from tuning
        
        Args:
            model_types: List of model types to evaluate (default: all)
            pairs: List of currency pairs to process (default: all)
            data_types: List of data types to use (default: 'selected')
        """
        print("\n" + "="*60)
        print("Evaluating Tuned Models")
        print("="*60)
        
        # Default is to evaluate all models for all pairs
        if model_types is None:
            model_types = ['LSTM', 'GRU', 'XGBoost', 'TFT']
        
        if pairs is None:
            pairs = self.config.CURRENCY_PAIRS
        
        if data_types is None:
            data_types = ['selected']
        
        # Load best hyperparameters
        best_params_file = os.path.join(self.config.HYPERPARAMS_PATH, "best_params_all.json")
        if not os.path.exists(best_params_file):
            print(f"Best parameters file not found: {best_params_file}")
            print("Please run hyperparameter tuning first.")
            return False
        
        with open(best_params_file, 'r') as f:
            best_params_all = json.load(f)
        
        # Need to have prepared model data
        if not hasattr(self, 'model_data') or not self.model_data:
            print("Model data not prepared. Running data preparation steps...")
            self.run_data_acquisition()
            self.run_feature_engineering()
            self.prepare_model_data()
        
        # Results storage
        results = {}
        
        # For each pair, model type, and data type, train and evaluate
        for pair in pairs:
            if pair not in results:
                results[pair] = {}
            
            for data_type in data_types:
                if data_type not in results[pair]:
                    results[pair][data_type] = {}
                
                if pair not in self.model_data or data_type not in self.model_data[pair]:
                    print(f"Data not available for {pair} {data_type}. Skipping.")
                    continue
                
                data = self.model_data[pair][data_type]
                
                # For each model type
                for model_type in model_types:
                    print(f"\nTraining {model_type} model for {pair} using {data_type} data with tuned hyperparameters...")
                    
                    # Get best parameters for this model and pair
                    if pair in best_params_all and model_type in best_params_all[pair] and data_type in best_params_all[pair][model_type]:
                        best_params = best_params_all[pair][model_type][data_type]
                        print(f"Using best parameters: {best_params}")
                    else:
                        print(f"No tuned parameters found for {model_type} on {pair} using {data_type} data. Using default parameters.")
                        best_params = None
                    
                    # Train model with best parameters
                    if model_type == 'LSTM':
                        model_instance = LSTMModel(self.config, best_params)
                        model, history = model_instance.train(
                            data['X_seq_train'], data['y_seq_train'], 
                            f"{pair}_{data_type}_{model_type}_tuned"
                        )
                        predictions = model_instance.predict(model, data['X_seq_test'], data['seq_scaler'], data['target_idx'])
                    
                    elif model_type == 'GRU':
                        model_instance = GRUModel(self.config, best_params)
                        model, history = model_instance.train(
                            data['X_seq_train'], data['y_seq_train'], 
                            f"{pair}_{data_type}_{model_type}_tuned"
                        )
                        predictions = model_instance.predict(model, data['X_seq_test'], data['seq_scaler'], data['target_idx'])
                    
                    elif model_type == 'XGBoost':
                        model_instance = XGBoostModel(self.config, best_params)
                        # Split data for validation
                        X_train, X_val, y_train, y_val = train_test_split(
                            data['X_tab_train'], data['y_tab_train'], 
                            test_size=0.2, shuffle=False
                        )
                        model = model_instance.train(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)]
                        )
                        predictions = model_instance.predict(model, data['X_tab_test'])
                        history = None
                    
                    elif model_type == 'TFT':
                        model_instance = TFTModel(self.config, best_params)
                        model, history = model_instance.train(
                            data['X_seq_train'], data['y_seq_train'], 
                            f"{pair}_{data_type}_{model_type}_tuned"
                        )
                        predictions = model_instance.predict(model, data['X_seq_test'], data['seq_scaler'], data['target_idx'])
                    
                    # Store results
                    actual_values = data['y_seq_test'] if model_type != 'XGBoost' else data['y_tab_test']
                    actual_dates = self.processed_data[pair][data_type]['test']['Time'].values[self.config.SEQUENCE_LENGTH:]
                    
                    results[pair][data_type][model_type] = {
                        'model': model,
                        'history': history,
                        'predictions': predictions,
                        'actual': actual_values,
                        'dates': actual_dates
                    }
                    
                    # Compare with untuned model
                    print("\nEvaluating tuned model performance...")
                    metrics = self.evaluator.calculate_metrics(actual_values, predictions)
                    trading_metrics = self.evaluator.calculate_trading_metrics(actual_values, predictions)
                    benchmark = self.evaluator.compare_with_benchmarks(actual_values, predictions)
                    
                    print("\nTuned model performance:")
                    print(f"  RMSE: {metrics['rmse']:.6f}")
                    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
                    print(f"  Annual Return: {trading_metrics['annual_return']:.2f}%")
        
        # Store results for further analysis
        self.tuned_results = results
        
        return True

    def run_visualization(self):
        """
        Create visualizations from results
        """
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)
        
        # Check if we have results to visualize
        if not self.results or not self.evaluation_results:
            print("Error: No results to visualize. Run training and evaluation first.")
            return False
        
        # Visualize results for each currency pair
        for pair in self.config.CURRENCY_PAIRS:
            if pair not in self.results:
                continue
                
            symbol = self.config.PAIR_SYMBOLS[pair]
            print(f"\nVisualizing results for {pair} [{symbol}]...")
            
            # Visualize results for each data type (E1, E2, E3, G1, G2, G3, J1, J2, J3)
            for idx, data_type in enumerate(['raw', 'enhanced', 'selected'], 1):
                model_identifier = f"{symbol}{idx}"
                if model_identifier not in self.results[pair]:
                    continue
                    
                print(f"Visualizing {data_type} results [{model_identifier}]...")
                
                # 1. Plot prediction comparison
                self.visualizer.plot_predictions_comparison(
                    self.results[pair][model_identifier], 
                    model_identifier, 
                    pair
                )
                
                # 2. Plot metrics comparison
                self.visualizer.plot_metrics_comparison(
                    self.evaluation_results[pair][model_identifier], 
                    model_identifier, 
                    pair
                )
            
            # 3. Plot data type comparison
            self.visualizer.plot_data_type_comparison(
                self.evaluation_results[pair], 
                symbol, 
                pair
            )
        
        # Visualize Bagging Approach results
        if 'bagging' in self.results:
            print("\nVisualizing Bagging results [B]...")
            
            # 1. Plot prediction comparison for each pair
            for pair in self.config.CURRENCY_PAIRS:
                if pair in self.results['bagging']:
                    self.visualizer.plot_predictions_comparison(
                        self.results['bagging'][pair], 
                        'B', 
                        pair
                    )
            
            # 2. Plot metrics comparison for each pair
            for pair in self.config.CURRENCY_PAIRS:
                if pair in self.evaluation_results['bagging']:
                    self.visualizer.plot_metrics_comparison(
                        self.evaluation_results['bagging'][pair], 
                        'B', 
                        pair
                    )
            
            # 3. Plot bagging comparison
            self.visualizer.plot_bagging_comparison(self.evaluation_results)
        
        return True
    
    def generate_reports(self):
        """
        Generate summary reports
        """
        print("\n" + "="*60)
        print("Generating Reports")
        print("="*60)
        
        # Check if we have results to report
        if not self.evaluation_results:
            print("Error: No evaluation results to report. Run evaluation first.")
            return False
        
        # Generate reports for each currency pair
        for pair in self.config.CURRENCY_PAIRS:
            if pair in self.evaluation_results:
                print(f"Creating summary report for {pair}...")
                self.reporter.create_pair_summary_report(self.evaluation_results[pair], pair)
        
        # Generate Bagging report
        if 'bagging' in self.evaluation_results:
            print("Creating Bagging summary report...")
            self.reporter.create_bagging_summary_report(self.evaluation_results['bagging'])
        
        # Generate overall summary report
        print("Creating overall summary report...")
        self.reporter.create_overall_summary_report(self.evaluation_results)
        
        return True
    
    def run_full_workflow(self, model_types=None, pairs=None):
        """
        Run the complete workflow
        
        Args:
            model_types: List of model types to train (default: all)
            pairs: List of currency pairs to process (default: all)
        """
        # Stage 1: Data Acquisition
        if not self.run_data_acquisition():
            print("Error in Stage 1: Data Acquisition. Stopping workflow.")
            return False
        
        # Stage 2: Feature Engineering
        if not self.run_feature_engineering():
            print("Error in Stage 2: Feature Engineering. Stopping workflow.")
            return False
        
        # Prepare data for models
        if not self.prepare_model_data():
            print("Error in preparing model data. Stopping workflow.")
            return False
        
        # Stage 3: Model Training
        if not self.run_model_training(model_types, pairs):
            print("Error in Stage 3: Model Training. Stopping workflow.")
            return False
        
        # Stage 4: Evaluation
        if not self.run_evaluation():
            print("Error in Stage 4: Evaluation. Stopping workflow.")
            return False
        
        # Visualization
        if not self.run_visualization():
            print("Error in visualization. Stopping workflow.")
            return False
        
        # Generate Reports
        if not self.generate_reports():
            print("Error in report generation. Stopping workflow.")
            return False
        
        print("\n" + "="*60)
        print("Workflow completed successfully!")
        print("="*60)
        
        return True


def main():
    """
    Main entry point for the Forex Prediction System
    """
    parser = argparse.ArgumentParser(description='Forex Prediction System')
    
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'data', 'features', 'train', 'tune', 'evaluate', 'visualize', 'report'],
                        help='Mode to run the program')
    
    parser.add_argument('--pair', type=str, default='all',
                        choices=['all', 'EURUSD', 'GBPUSD', 'USDJPY', 'bagging'],
                        help='Currency pair to process')
    
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'LSTM', 'GRU', 'XGBoost', 'TFT'],
                        help='Model type to use')
    
    parser.add_argument('--data_type', type=str, default='selected',
                        choices=['raw', 'enhanced', 'selected', 'all'],
                        help='Type of data to use')
    
    parser.add_argument('--trials', type=int, default=30,
                        help='Number of trials for hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Create Config
    config = Config
    
    # Process model types
    model_types = None
    if args.model != 'all':
        model_types = [args.model]
    
    # Process pairs
    pairs = None
    if args.pair != 'all':
        pairs = [args.pair]
    
    # Process data types
    data_types = None
    if args.data_type != 'all':
        data_types = [args.data_type]
    else:
        data_types = ['raw', 'enhanced', 'selected']
    
    # Create ForexPredictionSystem
    system = ForexPredictionSystem(config)
    
    # Run the selected mode
    if args.mode == 'all':
        system.run_full_workflow(model_types, pairs)
    
    elif args.mode == 'data':
        system.run_data_acquisition()
    
    elif args.mode == 'features':
        system.run_data_acquisition()
        system.run_feature_engineering()
    
    elif args.mode == 'train':
        system.run_data_acquisition()
        system.run_feature_engineering()
        system.prepare_model_data()
        system.run_model_training(model_types, pairs)
    
    elif args.mode == 'tune':
        # New mode for hyperparameter tuning
        system.run_data_acquisition()
        system.run_feature_engineering()
        system.prepare_model_data()
        system.run_hyperparameter_tuning(model_types, pairs, data_types, n_trials=args.trials)
    
    elif args.mode == 'evaluate':
        # Need to load trained models in this case
        print("Loading previously trained models is not implemented yet.")
        print("Please run with --mode=train first.")
    
    elif args.mode == 'visualize':
        # Need to load evaluation results in this case
        print("Loading previous evaluation results is not implemented yet.")
        print("Please run with --mode=train first.")
    
    elif args.mode == 'report':
        # Need to load evaluation results in this case
        print("Loading previous evaluation results is not implemented yet.")
        print("Please run with --mode=train first.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time/60:.2f} minutes")