"""
XGBoost Model for Forex Prediction
This module contains the implementation of the XGBoost model.
"""

import os
import numpy as np
import pandas as pd
import joblib
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
        else:
            print("XGBoost is available and will be used for training and prediction.")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
              early_stopping_rounds: Optional[int] = None,
              custom_params: Optional[Dict[str, Any]] = None,
              feature_names: Optional[List[str]] = None) -> Optional[xgb.XGBRegressor]:
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target values
            eval_set: Optional evaluation sets for early stopping
            early_stopping_rounds: Optional number of rounds with no improvement for early stopping
            custom_params: Optional dictionary of custom parameters to override default
            feature_names: Optional list of feature names
            
        Returns:
            Trained XGBoost model or None if XGBoost is not available
        """
        if not XGBOOST_AVAILABLE:
            print("XGBoost is not available. Skipping XGBoost model training.")
            return None
        
        # ตรวจสอบข้อมูลนำเข้า
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must not be None")
            
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train length ({len(X_train)}) does not match y_train length ({len(y_train)})")
            
        # ตรวจสอบและแก้ไข NaN
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            print("WARNING: NaN values detected in training data. Replacing with zeros.")
            X_train = np.nan_to_num(X_train, nan=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0)
            
        if np.isinf(X_train).any() or np.isinf(y_train).any():
            print("WARNING: Inf values detected in training data. Replacing with large values.")
            X_train = np.nan_to_num(X_train, posinf=1e10, neginf=-1e10)
            y_train = np.nan_to_num(y_train, posinf=1e10, neginf=-1e10)
            
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
            tree_method=params.get('tree_method', 'hist'),  # 'hist' is faster than 'exact'
            n_jobs=params.get('n_jobs', -1)  # Use all available cores
        )
        
        # Print information
        print(f"Training XGBoost model with {len(X_train)} samples, {X_train.shape[1]} features")
        print(f"Parameters: n_estimators={params.get('n_estimators')}, max_depth={params.get('max_depth')}, learning_rate={params.get('learning_rate')}")
        
        # If early_stopping_rounds is not provided, use config patience
        if early_stopping_rounds is None and eval_set is not None:
            early_stopping_rounds = self.config.PATIENCE
        
        # เตรียม kwargs สำหรับ fit
        fit_kwargs = {
            'X': X_train, 
            'y': y_train,
            'verbose': True
        }
        
        # ถ้ามี feature_names ให้ใช้
        if feature_names is not None:
            if len(feature_names) == X_train.shape[1]:
                fit_kwargs['feature_names'] = feature_names
            else:
                print(f"WARNING: feature_names length ({len(feature_names)}) does not match X_train columns ({X_train.shape[1]}). Ignoring feature_names.")
        
        # Add eval_set and early_stopping_rounds if provided
        if eval_set is not None:
            # ตรวจสอบและแก้ไข NaN, Inf ใน eval_set
            cleaned_eval_set = []
            for X_eval, y_eval in eval_set:
                if np.isnan(X_eval).any() or np.isnan(y_eval).any() or np.isinf(X_eval).any() or np.isinf(y_eval).any():
                    print("WARNING: NaN or Inf values detected in evaluation set. Cleaning data.")
                    X_eval_clean = np.nan_to_num(X_eval, nan=0.0, posinf=1e10, neginf=-1e10)
                    y_eval_clean = np.nan_to_num(y_eval, nan=0.0, posinf=1e10, neginf=-1e10)
                    cleaned_eval_set.append((X_eval_clean, y_eval_clean))
                else:
                    cleaned_eval_set.append((X_eval, y_eval))
                    
            fit_kwargs['eval_set'] = cleaned_eval_set
            fit_kwargs['eval_metric'] = 'rmse'
            
            if early_stopping_rounds is not None:
                fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
                print(f"Using early stopping with patience {early_stopping_rounds}")
        
        try:
            # เทรนโมเดล
            model.fit(**fit_kwargs)
            
            # Print best iteration if early stopping was used
            if eval_set is not None and hasattr(model, 'best_iteration'):
                print(f"Best iteration: {model.best_iteration}")
                
            return model
        except Exception as e:
            print(f"Error during XGBoost model training: {e}")
            return None
    
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
        
        # ตรวจสอบข้อมูลนำเข้า
        if X_test is None or len(X_test) == 0:
            return np.array([])
        
        # ตรวจสอบและแก้ไข NaN, Inf
        if np.isnan(X_test).any() or np.isinf(X_test).any():
            print("WARNING: NaN or Inf values detected in test data. Cleaning data.")
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
            
        print(f"Making predictions on {len(X_test)} samples")
        
        try:
            return model.predict(X_test)
        except Exception as e:
            print(f"Error during XGBoost prediction: {e}")
            return np.zeros(len(X_test))
    
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
        
        # สร้างโฟลเดอร์ถ้ายังไม่มี
        if not os.path.exists(self.config.MODELS_PATH):
            os.makedirs(self.config.MODELS_PATH)
            
        # บันทึกโมเดลในรูปแบบต่างๆ
        try:
            # บันทึกในรูปแบบ JSON (ใช้ได้กับ XGBoost เวอร์ชันใหม่)
            model_path_json = os.path.join(self.config.MODELS_PATH, f"{model_name}.json")
            model.save_model(model_path_json)
            print(f"XGBoost model saved to {model_path_json}")
            
            # บันทึกในรูปแบบ Pickle (เข้ากันได้กับหลายเวอร์ชัน)
            model_path_pkl = os.path.join(self.config.MODELS_PATH, f"{model_name}.pkl")
            joblib.dump(model, model_path_pkl)
            print(f"XGBoost model also saved to {model_path_pkl}")
            
        except Exception as e:
            print(f"Error saving XGBoost model: {e}")
            
            # ลองใช้วิธีอื่น
            try:
                model_path_pkl = os.path.join(self.config.MODELS_PATH, f"{model_name}.pkl")
                joblib.dump(model, model_path_pkl)
                print(f"XGBoost model saved to {model_path_pkl} using joblib")
            except Exception as e2:
                print(f"Error saving XGBoost model using joblib: {e2}")
    
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
        
        # ลองโหลดจากไฟล์ JSON ก่อน
        model_path_json = os.path.join(self.config.MODELS_PATH, f"{model_name}.json")
        if os.path.exists(model_path_json):
            try:
                model = xgb.XGBRegressor()
                model.load_model(model_path_json)
                print(f"Loaded XGBoost model from {model_path_json}")
                return model
            except Exception as e:
                print(f"Error loading XGBoost model from JSON: {e}")
        
        # ลองโหลดจากไฟล์ Pickle
        model_path_pkl = os.path.join(self.config.MODELS_PATH, f"{model_name}.pkl")
        if os.path.exists(model_path_pkl):
            try:
                model = joblib.load(model_path_pkl)
                print(f"Loaded XGBoost model from {model_path_pkl}")
                return model
            except Exception as e:
                print(f"Error loading XGBoost model from Pickle: {e}")
        
        print(f"No model file found for {model_name}")
        return None
    
    def get_feature_importance(self, model: Optional[xgb.XGBRegressor], 
                              feature_names: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
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
        
        try:
            # Get feature importance
            importance = model.feature_importances_
            
            # Ensure we have feature names
            if feature_names is None:
                # Try to get feature names from model
                if hasattr(model, 'feature_names_'):
                    feature_names = model.feature_names_
                else:
                    # Create default feature names
                    feature_names = [f'f{i}' for i in range(len(importance))]
            elif len(feature_names) != len(importance):
                print(f"WARNING: feature_names length ({len(feature_names)}) does not match feature importance length ({len(importance)}). Using default names.")
                feature_names = [f'f{i}' for i in range(len(importance))]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Print top 10 most important features
            print("\nTop 10 important features:")
            print(importance_df.head(10))
            
            return importance_df
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return None
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray,
                          param_grid: Optional[Dict[str, List[Any]]] = None,
                          n_trials: int = 10) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna
        
        Args:
            X_train: Training features
            y_train: Training target values
            X_val: Validation features
            y_val: Validation target values
            param_grid: Dictionary of parameter grid to search
            n_trials: Number of trials for optimization
            
        Returns:
            Dictionary of best parameters
        """
        if not XGBOOST_AVAILABLE:
            print("XGBoost is not available. Cannot tune hyperparameters.")
            return {}
            
        try:
            import optuna
            
            # Default parameter grid if not provided
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2]
                }
            
            # Define objective function
            def objective(trial):
                params = {
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
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)], 
                         early_stopping_rounds=20,
                         verbose=False)
                
                # Get RMSE on validation set
                preds = model.predict(X_val)
                rmse = np.sqrt(np.mean((y_val - preds) ** 2))
                
                return rmse
            
            # Create study
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            print(f"Best parameters: {best_params}")
            print(f"Best RMSE: {study.best_value:.6f}")
            
            return best_params
        
        except ImportError:
            print("Optuna is not available. Install with 'pip install optuna'.")
            return {}
        except Exception as e:
            print(f"Error during hyperparameter tuning: {e}")
            return {}
    
    def explain_predictions(self, model: Optional[xgb.XGBRegressor], X: np.ndarray, 
                         feature_names: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Explain model predictions using SHAP values
        
        Args:
            model: Trained XGBoost model
            X: Features to explain
            feature_names: Feature names
            
        Returns:
            Dictionary with SHAP values and explanations
        """
        if model is None or not XGBOOST_AVAILABLE:
            print("XGBoost model not available. Cannot explain predictions.")
            return None
            
        try:
            import shap
            
            # Ensure we have feature names
            if feature_names is None:
                if hasattr(model, 'feature_names_'):
                    feature_names = model.feature_names_
                else:
                    feature_names = [f'f{i}' for i in range(X.shape[1])]
            
            # Create explainer
            explainer = shap.TreeExplainer(model)
            
            # Sample data for explanation if too large
            if len(X) > 1000:
                print(f"Data too large ({len(X)} samples). Using 1000 samples for explanation.")
                indices = np.random.choice(len(X), 1000, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Get feature importance from SHAP values
            shap_importance = np.abs(shap_values).mean(0)
            shap_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Importance': shap_importance
            }).sort_values('SHAP_Importance', ascending=False)
            
            print("\nTop 10 features by SHAP importance:")
            print(shap_importance_df.head(10))
            
            return {
                'shap_values': shap_values,
                'shap_importance': shap_importance_df,
                'mean_shap_values': shap_values.mean(0),
                'explainer': explainer
            }
        
        except ImportError:
            print("SHAP is not available. Install with 'pip install shap'.")
            return None
        except Exception as e:
            print(f"Error explaining predictions: {e}")
            return None