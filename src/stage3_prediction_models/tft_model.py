"""
Temporal Fusion Transformer Model for Forex Prediction
This module contains a simple implementation of the Temporal Fusion Transformer model.
"""

import os
import numpy as np
from typing import Tuple, List, Dict, Optional, Any

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError


class TFTModel:
    """
    Simplified Temporal Fusion Transformer model for time series prediction
    
    Note: This is a simplified version of TFT, not the full implementation.
    For a full implementation, consider using libraries like pytorch-forecasting.
    """
    
    def __init__(self, config, params=None):
        """
        Initialize the TFT model with configuration parameters
        
        Args:
            config: Configuration object containing model parameters
            params: Optional dictionary of model hyperparameters
        """
        self.config = config
        self.params = params if params is not None else config.TFT_PARAMS
    
    def build_model(self, input_shape: Tuple[int, int], num_features: int) -> tf.keras.Model:
        """
        Build and compile the TFT model architecture
        
        Args:
            input_shape: Shape of the input data (sequence_length, num_features)
            num_features: Number of features in the input data
            
        Returns:
            Compiled TFT model
        """
        # อ่านพารามิเตอร์
        learning_rate = self.params.get('learning_rate', 0.001)
        attention_heads = self.params.get('attention_heads', 4)
        dropout = self.params.get('dropout', 0.1)
        hidden_units = self.params.get('hidden_units', 64)
        
        # Print model configuration
        print(f"\nBuilding TFT model with:")
        print(f"- Input shape: {input_shape}")
        print(f"- Features: {num_features}")
        print(f"- Attention heads: {attention_heads}")
        print(f"- Hidden units: {hidden_units}")
        print(f"- Dropout: {dropout}")
        
        # สร้างโมเดล
        # Input layer
        inputs = Input(shape=input_shape, name='input_layer')
        
        # Encoder (LSTM layer)
        x = LSTM(hidden_units, return_sequences=True, name='lstm_encoder')(inputs)
        x = Dropout(dropout, name='dropout_encoder')(x)
        
        # สร้างแบบจำลองของ Multi-head attention layer
        attention_outputs = []
        for i in range(attention_heads):
            # Simple attention mechanism
            attention = Dense(hidden_units, activation='tanh', name=f'attention_dense_{i}')(x)
            attention = Dense(1, activation='softmax', name=f'attention_weights_{i}')(attention)
            attention_output = tf.keras.layers.multiply([x, attention], name=f'attention_applied_{i}')
            attention_outputs.append(attention_output)
        
        # รวม attention heads
        if attention_heads > 1:
            x = concatenate(attention_outputs, name='concatenate_attention')
        else:
            x = attention_outputs[0]
        
        # Decoder (LSTM layer)
        x = LSTM(hidden_units, name='lstm_decoder')(x)
        x = Dropout(dropout, name='dropout_decoder')(x)
        
        # Output layer
        outputs = Dense(1, name='output_layer')(x)
        
        # สร้างโมเดล
        model = Model(inputs=inputs, outputs=outputs)
        
        # คอมไพล์โมเดล
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[RootMeanSquaredError()]
        )
        
        # Print model summary
        model.summary()
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, model_name: str, 
              callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """
        Train the TFT model
        
        Args:
            X_train: Training features (shape: [samples, sequence_length, features])
            y_train: Training target values
            model_name: Name for saving the model
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        # สร้างโมเดล
        num_features = X_train.shape[2]
        model = self.build_model(X_train.shape[1:], num_features)
        
        # ถ้าไม่มี callbacks กำหนดให้
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss', 
                    patience=self.config.PATIENCE, 
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath=os.path.join(self.config.MODELS_PATH, f"{model_name}.h5"),
                    save_best_only=True,
                    monitor='val_loss',
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.config.PATIENCE // 2,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        
        # เทรนโมเดล
        print(f"\nTraining TFT model with {len(X_train)} samples")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        history = model.fit(
            X_train, y_train,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            validation_split=self.config.VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def predict(self, model: tf.keras.Model, X_test: np.ndarray, 
                scaler: Any, target_idx: int) -> np.ndarray:
        """
        Generate predictions and inverse transform to original scale
        
        Args:
            model: Trained TFT model
            X_test: Test features
            scaler: Scaler object used for normalization
            target_idx: Index of target column in the original dataset
            
        Returns:
            Array of predictions in the original scale
        """
        # ทำนาย
        print(f"Making predictions with TFT model on {len(X_test)} samples")
        y_pred_scaled = model.predict(X_test)
        
        # สร้าง array สำหรับการแปลงกลับ
        dummy = np.zeros((len(y_pred_scaled), scaler.scale_.shape[0]))
        dummy[:, target_idx] = y_pred_scaled.flatten()
        
        # แปลงกลับเฉพาะคอลัมน์เป้าหมาย
        y_pred = scaler.inverse_transform(dummy)[:, target_idx]
        
        return y_pred