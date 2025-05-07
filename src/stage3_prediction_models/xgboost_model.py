"""
XGBoost Model for Forex Prediction
This module contains the implementation of the XGBoost model.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any, Union

# Import XGBoost with error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost is not available. XGBoost model will be skipped.")
    XGBOOST_AVAILABLE = False


class XGBoostModel:
    """
    XGBoost Gradient Boosting model for time series prediction
    """
    
    def __init__(self, config, params=None):
        """
        Initialize the XGBoost model with configuration parameters
        
        Args:
            config: Configuration object containing model parameters
            params: Optional dictionary of model hyperparameters
        """
        self.config = config
        self.params = params if params is not None else config.XGB_PARAMS
        
        if not XGBOOST_AVAILABLE:
            print("XGBoost is not available. Functionality will be limited.")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
          eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
          early_stopping_rounds: Optional[int] = None,
          custom_params: Optional[Dict[str, Any]] = None) -> Optional[xgb.XGBRegressor]:
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target values
            eval_set: Optional evaluation sets for early stopping
            early_stopping_rounds: Optional number of rounds with no improvement for early stopping
            custom_params: Optional dictionary of custom parameters to override default
            
        Returns:
            Trained XGBoost model or None if XGBoost is not available
        """
        if not XGBOOST_AVAILABLE:
            print("XGBoost is not available. Skipping XGBoost model training.")
            return None
            
        # Merge default parameters with custom parameters if provided
        params = self.params.copy()
        if custom_params is not None:
            params.update(custom_params)
                
        # สร้างโมเดล
        model = xgb.XGBRegressor(
            n_estimators=params.get('n_estimators', 500),
            max_depth=params.get('max_depth', 8),
            learning_rate=params.get('learning_rate', 0.01),
            gamma=params.get('gamma', 0.1),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            min_child_weight=params.get('min_child_weight', 3),
            reg_alpha=params.get('reg_alpha', 0.1),
            reg_lambda=params.get('reg_lambda', 1),
            random_state=params.get('random_state', 42),
            tree_method='hist',  # ใช้ histogram-based algorithm ซึ่งเร็วกว่า
            n_jobs=-1  # Use all available cores
        )
        
        # Print information
        print(f"Training XGBoost model with {len(X_train)} samples, {X_train.shape[1]} features")
        print(f"Parameters: n_estimators={params.get('n_estimators')}, max_depth={params.get('max_depth')}, learning_rate={params.get('learning_rate')}")
        
        # If early_stopping_rounds is not provided, use config patience
        if early_stopping_rounds is None and eval_set is not None:
            early_stopping_rounds = self.config.PATIENCE
        
        # เทรนโมเดล
        fit_kwargs = {
            'X': X_train, 
            'y': y_train,
            'verbose': True
        }
        
        # Add eval_set and early_stopping_rounds if provided
        if eval_set is not None:
            fit_kwargs['eval_set'] = eval_set
            fit_kwargs['eval_metric'] = 'rmse'
            
            if early_stopping_rounds is not None:
                fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
                print(f"Using early stopping with patience {early_stopping_rounds}")
        
        model.fit(**fit_kwargs)
        
        # Print best iteration if early stopping was used
        if eval_set is not None and hasattr(model, 'best_iteration'):
            print(f"Best iteration: {model.best_iteration}")
        
        return model
    
    def predict(self, model: Optional[xgb.XGBRegressor], X_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model
        
        Args:
            model: Trained XGBoost model
            X_test: Test features
            
        Returns:
            Array of predictions
        """
        if model is None or not XGBOOST_AVAILABLE:
            print("XGBoost model not available. Returning zeros.")
            return np.zeros(len(X_test))
            
        print(f"Making predictions on {len(X_test)} samples")
        return model.predict(X_test)
    
    def save_model(self, model: Optional[xgb.XGBRegressor], model_name: str) -> None:
        """
        Save XGBoost model to disk
        
        Args:
            model: Trained XGBoost model
            model_name: Name for saving the model
        """
        if model is None or not XGBOOST_AVAILABLE:
            print("XGBoost model not available. Cannot save model.")
            return
            
        model_path = os.path.join(self.config.MODELS_PATH, f"{model_name}.json")
        model.save_model(model_path)
        print(f"XGBoost model saved to {model_path}")
    
    def load_model(self, model_name: str) -> Optional[xgb.XGBRegressor]:
        """
        Load XGBoost model from disk
        
        Args:
            model_name: Name of the saved model
            
        Returns:
            Loaded XGBoost model
        """
        if not XGBOOST_AVAILABLE:
            print("XGBoost is not available. Cannot load model.")
            return None
            
        model_path = os.path.join(self.config.MODELS_PATH, f"{model_name}.json")
        model = xgb.XGBRegressor()
        try:
            model.load_model(model_path)
            print(f"Loaded XGBoost model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading XGBoost model: {e}")
            return None
    
    def get_feature_importance(self, model: Optional[xgb.XGBRegressor], 
                              feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the model
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance or None if model is not available
        """
        if model is None or not XGBOOST_AVAILABLE:
            print("XGBoost model not available. Cannot get feature importance.")
            return None
            
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Print top 10 most important features
        print("\nTop 10 important features:")
        print(importance_df.head(10))
        
        return importance_df