"""
Temporal Fusion Transformer Model for Forex Prediction
This module contains a simple implementation of the Temporal Fusion Transformer model.
"""

import os
import numpy as np
from typing import Tuple, List, Dict, Optional, Any

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, concatenate, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D
)
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
        hidden_continuous_size = self.params.get('hidden_continuous_size', 32)
        
        # Print model configuration
        print(f"\nBuilding TFT model with:")
        print(f"- Input shape: {input_shape}")
        print(f"- Features: {num_features}")
        print(f"- Attention heads: {attention_heads}")
        print(f"- Hidden units: {hidden_units}")
        print(f"- Hidden continuous size: {hidden_continuous_size}")
        print(f"- Dropout: {dropout}")
        
        # สร้างโมเดล
        # Input layer
        inputs = Input(shape=input_shape, name='input_layer')
        
        # Variable selection (simplified)
        vs_layer = Dense(hidden_continuous_size, activation='relu', name='variable_selection')(inputs)
        vs_layer = Dropout(dropout, name='vs_dropout')(vs_layer)
        
        # Encoder (LSTM layer)
        encoder = LSTM(hidden_units, return_sequences=True, name='lstm_encoder')(vs_layer)
        encoder = Dropout(dropout, name='encoder_dropout')(encoder)
        
        # Self-attention
        # Layer normalization
        norm_layer = LayerNormalization(epsilon=1e-6, name='layer_norm_1')(encoder)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=hidden_units // attention_heads,
            name='multi_head_attention'
        )(norm_layer, norm_layer, norm_layer)
        
        # Add & Norm (residual connection)
        attention_output = tf.keras.layers.Add(name='residual_1')([attention_output, encoder])
        attention_output = LayerNormalization(epsilon=1e-6, name='layer_norm_2')(attention_output)
        
        # Position-wise Feed-Forward Network
        ffn = Dense(hidden_units * 4, activation='relu', name='ffn_1')(attention_output)
        ffn = Dropout(dropout, name='ffn_dropout_1')(ffn)
        ffn = Dense(hidden_units, name='ffn_2')(ffn)
        ffn = Dropout(dropout, name='ffn_dropout_2')(ffn)
        
        # Add & Norm (another residual connection)
        ffn_output = tf.keras.layers.Add(name='residual_2')([ffn, attention_output])
        ffn_output = LayerNormalization(epsilon=1e-6, name='layer_norm_3')(ffn_output)
        
        # Temporal Fusion Decoder: Simplified using an LSTM
        decoder = LSTM(hidden_units, name='lstm_decoder')(ffn_output)
        decoder = Dropout(dropout, name='decoder_dropout')(decoder)
        
        # Final MLP layers
        outputs = Dense(hidden_units // 2, activation='relu', name='final_dense_1')(decoder)
        outputs = Dropout(dropout / 2, name='final_dropout')(outputs)
        outputs = Dense(1, name='output_layer')(outputs)
        
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
              callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
              epochs: Optional[int] = None,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """
        Train the TFT model
        
        Args:
            X_train: Training features (shape: [samples, sequence_length, features])
            y_train: Training target values
            model_name: Name for saving the model
            callbacks: Optional list of Keras callbacks
            epochs: Optional number of epochs (default: use config.EPOCHS)
            validation_data: Optional tuple of (X_val, y_val) for validation
            
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
                    factor=0.5,  # Reduce by half
                    patience=5,  # Check sooner
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        
        # กำหนดจำนวน epochs
        if epochs is None:
            epochs = self.config.EPOCHS
        
        # เตรียม kwargs สำหรับ model.fit
        fit_kwargs = {
            'x': X_train,
            'y': y_train,
            'epochs': epochs,
            'batch_size': self.config.BATCH_SIZE,
            'callbacks': callbacks,
            'verbose': 1
        }
        
        # ใช้ validation_data ถ้ามี หรือใช้ validation_split จาก config
        if validation_data is not None:
            fit_kwargs['validation_data'] = validation_data
        else:
            fit_kwargs['validation_split'] = self.config.VALIDATION_SPLIT
        
        # เทรนโมเดล
        print(f"\nTraining TFT model with {len(X_train)} samples")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        history = model.fit(**fit_kwargs)
        
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