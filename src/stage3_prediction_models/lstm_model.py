"""
LSTM Model for Forex Prediction
This module contains the implementation of the LSTM model.
"""

import os
import numpy as np
from typing import Tuple, List, Dict, Optional, Any

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.regularizers import l1_l2


class LSTMModel:
    """
    Long Short-Term Memory Neural Network model for time series prediction
    """
    
    def __init__(self, config, params=None):
        """
        Initialize the LSTM model with configuration parameters
        
        Args:
            config: Configuration object containing model parameters
            params: Optional dictionary of model hyperparameters
        """
        self.config = config
        self.params = params if params is not None else config.LSTM_PARAMS
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build and compile the LSTM model architecture
        
        Args:
            input_shape: Shape of the input data (sequence_length, num_features)
            
        Returns:
            Compiled LSTM model
        """
        # อ่านพารามิเตอร์
        units_layer1 = self.params.get('units_layer1', 100)
        units_layer2 = self.params.get('units_layer2', 50)
        dropout1 = self.params.get('dropout1', 0.3)
        dropout2 = self.params.get('dropout2', 0.3)
        recurrent_dropout = self.params.get('recurrent_dropout', 0.2)
        l1_reg = self.params.get('l1_reg', 0.0001)
        l2_reg = self.params.get('l2_reg', 0.0001)
        
        # สร้างโมเดล
        model = Sequential([
            LSTM(
                units_layer1, 
                return_sequences=True, 
                input_shape=input_shape,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name='lstm_layer1'
            ),
            Dropout(dropout1, name='dropout_layer1'),
            LSTM(
                units_layer2, 
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name='lstm_layer2'
            ),
            Dropout(dropout2, name='dropout_layer2'),
            Dense(1, name='output_layer')
        ])
        
        # คอมไพล์โมเดล
        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.001))
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
        Train the LSTM model
        
        Args:
            X_train: Training features (shape: [samples, sequence_length, features])
            y_train: Training target values
            model_name: Name for saving the model
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        # สร้างโมเดล
        model = self.build_model(X_train.shape[1:])
        
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
            model: Trained LSTM model
            X_test: Test features
            scaler: Scaler object used for normalization
            target_idx: Index of target column in the original dataset
            
        Returns:
            Array of predictions in the original scale
        """
        # ทำนาย
        y_pred_scaled = model.predict(X_test)
        
        # สร้าง array สำหรับการแปลงกลับ
        dummy = np.zeros((len(y_pred_scaled), scaler.scale_.shape[0]))
        dummy[:, target_idx] = y_pred_scaled.flatten()
        
        # แปลงกลับเฉพาะคอลัมน์เป้าหมาย
        y_pred = scaler.inverse_transform(dummy)[:, target_idx]
        
        return y_pred