"""
Ensemble Model for Forex Prediction
This module contains lightweight ensemble techniques for combining multiple models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LightweightEnsemble:
    """
    A lightweight ensemble model that combines predictions from multiple models
    with an emphasis on directional accuracy.
    """
    
    def __init__(self, config):
        """
        Initialize the ensemble model
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.models = {}
        self.weights = {}
        self.model_types = {}  # เก็บประเภทของโมเดลแต่ละตัว (sequence หรือ tabular)
    
    def add_model(self, name: str, model: Any, weight: float = 1.0, model_type: str = "sequence"):
        """
        Add a model to the ensemble
        
        Args:
            name: Name of the model
            model: Model object
            weight: Weight of the model in the ensemble
            model_type: Type of model ('sequence' or 'tabular')
        """
        if name in self.models:
            print(f"Warning: Model '{name}' already exists in the ensemble. Overwriting.")
            
        self.models[name] = model
        self.weights[name] = weight
        self.model_types[name] = model_type
        
        print(f"Added model '{name}' with weight {weight} and type '{model_type}' to the ensemble")
    
    def set_weights(self, weights_dict: Dict[str, float]):
        """
        Set weights for existing models
        
        Args:
            weights_dict: Dictionary of model names and weights
        """
        for name, weight in weights_dict.items():
            if name in self.models:
                self.weights[name] = weight
                print(f"Updated weight for model '{name}' to {weight}")
            else:
                print(f"Warning: Model '{name}' not found in the ensemble. Skipping.")
    
    def normalize_weights(self):
        """
        Normalize weights to sum to 1.0
        """
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            # If all weights are 0, set equal weights
            for name in self.weights:
                self.weights[name] = 1.0 / len(self.weights)
            print("All weights were 0. Set to equal weights.")
        else:
            # Normalize weights
            for name in self.weights:
                self.weights[name] /= total_weight
            print("Normalized weights to sum to 1.0")
    
    def predict(self, X_dict: Dict[str, Any]) -> np.ndarray:
        """
        Generate predictions using the ensemble
        
        Args:
            X_dict: Dictionary of inputs for different model types
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in the ensemble. Add models first.")
            
        # ตรวจสอบข้อมูลที่จำเป็น
        required_keys = ['sequence', 'tabular', 'seq_scaler', 'target_idx']
        for key in required_keys:
            if key not in X_dict:
                raise ValueError(f"Missing required key '{key}' in X_dict")
        
        all_preds = {}
        
        # ทำนายจากแต่ละโมเดล
        for model_name, model in self.models.items():
            model_type = self.model_types.get(model_name, "sequence")
            
            if model_type == "sequence":
                # ใช้ข้อมูล sequence สำหรับโมเดล sequence
                X = X_dict['sequence']
                scaler = X_dict['seq_scaler']
                target_idx = X_dict['target_idx']
                
                # เลือกฟังก์ชันทำนายตามประเภทของโมเดล
                if model_name.startswith(('LSTM', 'GRU', 'TFT')):
                    all_preds[model_name] = model.predict(model, X, scaler, target_idx)
                else:
                    # สำหรับโมเดลอื่นๆ ที่ใช้ข้อมูล sequence
                    all_preds[model_name] = model.predict(X)
            else:
                # ใช้ข้อมูล tabular สำหรับโมเดล tabular
                X = X_dict['tabular']
                
                # สำหรับโมเดล XGBoost
                if model_name.startswith('XGBoost'):
                    all_preds[model_name] = model.predict(model, X)
                else:
                    # สำหรับโมเดล tabular อื่นๆ
                    all_preds[model_name] = model.predict(X)
        
        # ตรวจสอบว่ามีการทำนายจากโมเดลอย่างน้อยหนึ่งตัว
        if not all_preds:
            raise ValueError("No predictions generated from any model.")
            
        # สร้าง weighted average
        sample_pred = list(all_preds.values())[0]
        weighted_pred = np.zeros_like(sample_pred)
        
        for model_name, preds in all_preds.items():
            if preds.shape != weighted_pred.shape:
                print(f"Warning: Shape mismatch for model '{model_name}'. Expected {weighted_pred.shape}, got {preds.shape}. Skipping.")
                continue
                
            weighted_pred += self.weights[model_name] * preds
        
        return weighted_pred
    
    def predict_direction(self, X_dict: Dict[str, Any]) -> np.ndarray:
        """
        Predict direction using weighted voting
        
        Args:
            X_dict: Dictionary of test data for each model type
            
        Returns:
            Array of predicted directions (1: up, -1: down)
        """
        # ทำนายค่าจริง
        predictions = self.predict(X_dict)
        
        # คำนวณทิศทาง
        directions = np.sign(np.diff(predictions))
        
        return directions
    
    def evaluate(self, X_dict: Dict[str, Any], y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the ensemble's performance
        
        Args:
            X_dict: Dictionary of test data for each model type
            y_true: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # ทำนายค่า
        y_pred = self.predict(X_dict)
        
        # คำนวณ metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # คำนวณ Directional Accuracy
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'predictions': y_pred
        }
    
    def optimize_weights(self, X_dict: Dict[str, Any], y_true: np.ndarray, 
                        metric: str = 'directional_accuracy', n_trials: int = 100) -> Dict[str, float]:
        """
        Optimize ensemble weights using a simple grid search
        
        Args:
            X_dict: Dictionary of test data for each model type
            y_true: True target values
            metric: Metric to optimize ('rmse', 'mae', 'directional_accuracy')
            n_trials: Number of random trials
            
        Returns:
            Dictionary of optimized weights
        """
        if len(self.models) == 0:
            raise ValueError("No models in the ensemble. Add models first.")
            
        if len(self.models) == 1:
            print("Only one model in the ensemble. Setting weight to 1.0.")
            model_name = list(self.models.keys())[0]
            return {model_name: 1.0}
        
        print(f"Optimizing weights for {len(self.models)} models using {n_trials} trials...")
        
        # ทำนายจากแต่ละโมเดล
        model_predictions = {}
        for model_name, model in self.models.items():
            model_type = self.model_types.get(model_name, "sequence")
            
            if model_type == "sequence":
                # ใช้ข้อมูล sequence สำหรับโมเดล sequence
                X = X_dict['sequence']
                scaler = X_dict['seq_scaler']
                target_idx = X_dict['target_idx']
                
                # เลือกฟังก์ชันทำนายตามประเภทของโมเดล
                if model_name.startswith(('LSTM', 'GRU', 'TFT')):
                    model_predictions[model_name] = model.predict(model, X, scaler, target_idx)
                else:
                    # สำหรับโมเดลอื่นๆ ที่ใช้ข้อมูล sequence
                    model_predictions[model_name] = model.predict(X)
            else:
                # ใช้ข้อมูล tabular สำหรับโมเดล tabular
                X = X_dict['tabular']
                
                # สำหรับโมเดล XGBoost
                if model_name.startswith('XGBoost'):
                    model_predictions[model_name] = model.predict(model, X)
                else:
                    # สำหรับโมเดล tabular อื่นๆ
                    model_predictions[model_name] = model.predict(X)
        
        # หา best weights
        best_weights = None
        best_score = float('inf') if metric in ['rmse', 'mae'] else float('-inf')
        
        for _ in range(n_trials):
            # สุ่ม weights
            weights = np.random.random(len(self.models))
            weights /= weights.sum()  # normalize
            
            # สร้าง weighted predictions
            weighted_pred = np.zeros_like(y_true)
            for i, (model_name, preds) in enumerate(model_predictions.items()):
                weighted_pred += weights[i] * preds
            
            # ประเมินผล
            if metric == 'rmse':
                score = np.sqrt(mean_squared_error(y_true, weighted_pred))
                is_better = score < best_score
            elif metric == 'mae':
                score = mean_absolute_error(y_true, weighted_pred)
                is_better = score < best_score
            elif metric == 'directional_accuracy':
                true_direction = np.sign(np.diff(y_true))
                pred_direction = np.sign(np.diff(weighted_pred))
                score = np.mean(true_direction == pred_direction) * 100
                is_better = score > best_score
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # อัปเดต best weights ถ้าได้ผลลัพธ์ที่ดีขึ้น
            if is_better:
                best_score = score
                best_weights = weights
                
                if metric in ['rmse', 'mae']:
                    print(f"New best {metric}: {best_score:.6f}")
                else:
                    print(f"New best {metric}: {best_score:.2f}%")
        
        # อัปเดต weights ในโมเดล
        optimized_weights = {}
        for i, model_name in enumerate(self.models.keys()):
            self.weights[model_name] = best_weights[i]
            optimized_weights[model_name] = best_weights[i]
            
        print(f"Optimized weights: {optimized_weights}")
        print(f"Best {metric}: {best_score}")
        
        return optimized_weights


class StackingEnsemble:
    """
    A stacking ensemble that uses predictions from base models as features
    for a meta-model.
    """
    
    def __init__(self, config):
        """
        Initialize the stacking ensemble
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.base_models = {}
        self.meta_model = None
        self.model_types = {}  # เก็บประเภทของโมเดลแต่ละตัว (sequence หรือ tabular)
    
    def add_base_model(self, name: str, model: Any, model_type: str = "sequence"):
        """
        Add a base model to the ensemble
        
        Args:
            name: Name of the model
            model: Model object
            model_type: Type of model ('sequence' or 'tabular')
        """
        if name in self.base_models:
            print(f"Warning: Model '{name}' already exists in the ensemble. Overwriting.")
            
        self.base_models[name] = model
        self.model_types[name] = model_type
        
        print(f"Added base model '{name}' with type '{model_type}' to the stacking ensemble")
    
    def set_meta_model(self, model: Any):
        """
        Set the meta-model for the stacking ensemble
        
        Args:
            model: Meta-model object
        """
        self.meta_model = model
        print(f"Set meta-model for the stacking ensemble: {type(model).__name__}")
    
    def fit(self, X_train_dict: Dict[str, Any], y_train: np.ndarray, 
            X_val_dict: Dict[str, Any], y_val: np.ndarray):
        """
        Train the stacking ensemble
        
        Args:
            X_train_dict: Dictionary of training data for different model types
            y_train: Training target values
            X_val_dict: Dictionary of validation data for different model types
            y_val: Validation target values
        """
        if not self.base_models:
            raise ValueError("No base models in the ensemble. Add base models first.")
            
        if self.meta_model is None:
            raise ValueError("No meta-model set. Set a meta-model first.")
        
        # ทำนายจากแต่ละโมเดลบน validation set
        base_predictions = np.zeros((len(y_val), len(self.base_models)))
        
        for i, (model_name, model) in enumerate(self.base_models.items()):
            model_type = self.model_types.get(model_name, "sequence")
            
            if model_type == "sequence":
                # ใช้ข้อมูล sequence สำหรับโมเดล sequence
                X_train = X_train_dict['sequence']
                X_val = X_val_dict['sequence']
                scaler = X_train_dict['seq_scaler']
                target_idx = X_train_dict['target_idx']
                
                # เลือกฟังก์ชันทำนายตามประเภทของโมเดล
                if model_name.startswith(('LSTM', 'GRU', 'TFT')):
                    # เทรนโมเดล
                    trained_model, _ = model.train(X_train, y_train, f"{model_name}_base")
                    # ทำนาย
                    base_predictions[:, i] = model.predict(trained_model, X_val, scaler, target_idx)
                else:
                    # สำหรับโมเดลอื่นๆ ที่ใช้ข้อมูล sequence
                    model.fit(X_train, y_train)
                    base_predictions[:, i] = model.predict(X_val)
            else:
                # ใช้ข้อมูล tabular สำหรับโมเดล tabular
                X_train = X_train_dict['tabular']
                X_val = X_val_dict['tabular']
                
                # สำหรับโมเดล XGBoost
                if model_name.startswith('XGBoost'):
                    # เทรนโมเดล
                    trained_model = model.train(X_train, y_train)
                    # ทำนาย
                    base_predictions[:, i] = model.predict(trained_model, X_val)
                else:
                    # สำหรับโมเดล tabular อื่นๆ
                    model.fit(X_train, y_train)
                    base_predictions[:, i] = model.predict(X_val)
        
        # เทรนเมตาโมเดล
        self.meta_model.fit(base_predictions, y_val)
        print(f"Stacking ensemble trained with {len(self.base_models)} base models")
    
    def predict(self, X_dict: Dict[str, Any]) -> np.ndarray:
        """
        Generate predictions using the stacking ensemble
        
        Args:
            X_dict: Dictionary of inputs for different model types
            
        Returns:
            Stacking ensemble predictions
        """
        if not self.base_models:
            raise ValueError("No base models in the ensemble. Add base models first.")
            
        if self.meta_model is None:
            raise ValueError("No meta-model set. Set a meta-model first.")
            
        # ตรวจสอบข้อมูลที่จำเป็น
        required_keys = ['sequence', 'tabular', 'seq_scaler', 'target_idx']
        for key in required_keys:
            if key not in X_dict:
                raise ValueError(f"Missing required key '{key}' in X_dict")
        
        # ทำนายจากแต่ละโมเดล
        base_predictions = np.zeros((X_dict['sequence'].shape[0], len(self.base_models)))
        
        for i, (model_name, model) in enumerate(self.base_models.items()):
            model_type = self.model_types.get(model_name, "sequence")
            
            if model_type == "sequence":
                # ใช้ข้อมูล sequence สำหรับโมเดล sequence
                X = X_dict['sequence']
                scaler = X_dict['seq_scaler']
                target_idx = X_dict['target_idx']
                
                # เลือกฟังก์ชันทำนายตามประเภทของโมเดล
                if model_name.startswith(('LSTM', 'GRU', 'TFT')):
                    base_predictions[:, i] = model.predict(model, X, scaler, target_idx)
                else:
                    # สำหรับโมเดลอื่นๆ ที่ใช้ข้อมูล sequence
                    base_predictions[:, i] = model.predict(X)
            else:
                # ใช้ข้อมูล tabular สำหรับโมเดล tabular
                X = X_dict['tabular']
                
                # สำหรับโมเดล XGBoost
                if model_name.startswith('XGBoost'):
                    base_predictions[:, i] = model.predict(model, X)
                else:
                    # สำหรับโมเดล tabular อื่นๆ
                    base_predictions[:, i] = model.predict(X)
        
        # ทำนายผลลัพธ์สุดท้ายโดยใช้เมตาโมเดล
        final_predictions = self.meta_model.predict(base_predictions)
        
        return final_predictions