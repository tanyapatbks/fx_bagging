# สร้างไฟล์ใหม่ src/ensemble_model.py
"""
Ensemble Model for Forex Prediction
This module contains lightweight ensemble techniques for combining multiple models.
"""

import numpy as np
from typing import Dict, Any, List

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
    
    def add_model(self, name, model, weight=1.0):
        """
        Add a model to the ensemble
        
        Args:
            name: Name of the model
            model: Model object
            weight: Weight of the model in the ensemble
        """
        self.models[name] = model
        self.weights[name] = weight
    
    def set_weights(self, weights_dict):
        """
        Set weights for existing models
        
        Args:
            weights_dict: Dictionary of model names and weights
        """
        for name, weight in weights_dict.items():
            if name in self.models:
                self.weights[name] = weight
    
    def create_lightweight_ensemble(self, X_dict, weights=None):
        """
        Create a simple weighted ensemble prediction
        
        Args:
            X_dict: Dictionary of inputs for different model types
            weights: Optional dictionary of model weights
            
        Returns:
            Ensemble predictions
        """
        if weights is None:
            # Default to currently set weights
            weights = self.weights
        
        all_preds = {}
        for model_name, model in self.models.items():
            # Get right input data for this model
            if model_name.startswith(('LSTM', 'GRU', 'TFT')):
                X = X_dict['sequence']
                scaler = X_dict['seq_scaler']
                target_idx = X_dict['target_idx']
                all_preds[model_name] = model.predict(X, scaler, target_idx)
            else:  # XGBoost
                X = X_dict['tabular']
                all_preds[model_name] = model.predict(X)
        
        # Create weighted average
        weighted_pred = np.zeros_like(list(all_preds.values())[0])
        for model_name, preds in all_preds.items():
            weighted_pred += weights[model_name] * preds
        
        return weighted_pred
    
    def predict_direction(self, X_dict):
        """
        Predict direction using weighted voting
        
        Args:
            X_dict: Dictionary of test data for each model type
            
        Returns:
            Array of predicted directions (1: up, -1: down)
        """
        predictions = {}
        directions = {}
        
        # Collect predictions from each model
        for model_name, model in self.models.items():
            if model_name.startswith(('LSTM', 'GRU', 'TFT')):
                # For sequence models
                X = X_dict['sequence']
                scaler = X_dict['seq_scaler']
                target_idx = X_dict['target_idx']
                predictions[model_name] = model.predict(X, scaler, target_idx)
            elif model_name.startswith('XGBoost'):
                # For tabular models
                X = X_dict['tabular']
                predictions[model_name] = model.predict(X)
            
            # Calculate directions
            directions[model_name] = np.sign(np.diff(predictions[model_name]))
        
        # Weighted vote on directions
        weighted_votes = np.zeros(directions[list(directions.keys())[0]].shape)
        
        for model_name, direction in directions.items():
            weighted_votes += direction * self.weights[model_name]
        
        # Final direction based on majority vote
        final_direction = np.sign(weighted_votes)
        
        return final_direction