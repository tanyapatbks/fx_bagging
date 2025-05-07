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
        สร้างและคอมไพล์โมเดล LSTM พร้อมกลไก Attention
        
        Args:
            input_shape: รูปร่างของข้อมูลนำเข้า (sequence_length, num_features)
            
        Returns:
            โมเดล LSTM ที่คอมไพล์แล้ว
        """
        # อ่านพารามิเตอร์
        units_layer1 = self.params.get('units_layer1', 128)
        units_layer2 = self.params.get('units_layer2', 64)
        dropout1 = self.params.get('dropout1', 0.3)
        dropout2 = self.params.get('dropout2', 0.3)
        recurrent_dropout = self.params.get('recurrent_dropout', 0.1)
        l1_reg = self.params.get('l1_reg', 0.0001)
        l2_reg = self.params.get('l2_reg', 0.0005)
        attention_units = self.params.get('attention_units', 32)
        
        # สร้างโมเดล
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # LSTM layer 1
        lstm1 = tf.keras.layers.LSTM(
            units_layer1, 
            return_sequences=True, 
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
            name='lstm_layer1'
        )(inputs)
        lstm1 = tf.keras.layers.Dropout(dropout1, name='dropout_layer1')(lstm1)
        
        # Attention mechanism
        attention = tf.keras.layers.Dense(attention_units, activation='tanh', name='attention_dense')(lstm1)
        attention = tf.keras.layers.Dense(1, activation='softmax', name='attention_weights')(attention)
        context_vector = tf.keras.layers.Multiply(name='attention_multiply')([lstm1, attention])
        context_vector = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='attention_sum')(context_vector)
        
        # LSTM layer 2 (optional, can use the context vector directly)
        if units_layer2 > 0:
            lstm2 = tf.keras.layers.LSTM(
                units_layer2, 
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                name='lstm_layer2'
            )(tf.keras.layers.Reshape((1, -1))(context_vector))
            lstm2 = tf.keras.layers.Dropout(dropout2, name='dropout_layer2')(lstm2)
            outputs = tf.keras.layers.Dense(1, name='output_layer')(lstm2)
        else:
            outputs = tf.keras.layers.Dense(1, name='output_layer')(context_vector)
        
        # สร้างโมเดล
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        # คอมไพล์โมเดล
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.get('learning_rate', 0.0005))
        model.compile(
            optimizer=optimizer, 
            loss='mse',
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        
        # Print model summary
        model.summary()
        
        return model
    
    def build_multi_horizon_attention(self, inputs, sequence_length, feature_dim):
        # Attention ระยะสั้น (ไม่กี่ชั่วโมงที่ผ่านมา)
        attention_short = tf.keras.layers.Dense(32, activation='tanh')(inputs[:, -6:, :])
        attention_short = tf.keras.layers.Dense(1, activation='softmax')(attention_short)
        context_short = tf.keras.layers.Multiply()([inputs[:, -6:, :], attention_short])
        context_short = tf.reduce_sum(context_short, axis=1)
        
        # Attention ระยะกลาง (กลางลำดับ)
        mid_start = max(0, sequence_length - 24)
        mid_end = sequence_length - 6
        attention_mid = tf.keras.layers.Dense(32, activation='tanh')(inputs[:, mid_start:mid_end, :])
        attention_mid = tf.keras.layers.Dense(1, activation='softmax')(attention_mid)
        context_mid = tf.keras.layers.Multiply()([inputs[:, mid_start:mid_end, :], attention_mid])
        context_mid = tf.reduce_sum(context_mid, axis=1)
        
        # Attention ระยะยาว (ลำดับทั้งหมด)
        attention_long = tf.keras.layers.Dense(32, activation='tanh')(inputs)
        attention_long = tf.keras.layers.Dense(1, activation='softmax')(attention_long)
        context_long = tf.keras.layers.Multiply()([inputs, attention_long])
        context_long = tf.reduce_sum(context_long, axis=1)
        
        # รวมบริบท
        combined_context = tf.keras.layers.Concatenate()([context_short, context_mid, context_long])
        return combined_context

    # ในการเตรียมข้อมูลของคุณ เพิ่มการจัดกลุ่มอย่างง่ายตามรูปแบบการเคลื่อนไหวของราคา
    def create_sequence_aware_batches(X, y, batch_size=32):
        # คำนวณลายเซ็นรูปแบบอย่างง่ายสำหรับแต่ละลำดับ (การเปลี่ยนทิศทาง)
        signatures = []
        for seq in X:
            # นับการเปลี่ยนทิศทางในราคาปิด
            close_idx = target_idx  # สมมติว่านี่คือดัชนีของราคาปิด
            closes = seq[:, close_idx]
            directions = np.sign(np.diff(closes))
            direction_changes = np.sum(np.abs(np.diff(directions)))
            signatures.append(direction_changes)
        
        # เรียงตามลายเซ็น
        sort_idx = np.argsort(signatures)
        return X[sort_idx], y[sort_idx]

    def train(self, X_train: np.ndarray, y_train: np.ndarray, model_name: str, 
          callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
          epochs: Optional[int] = None,
          validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """
        Train the LSTM model
        
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
        history = model.fit(**fit_kwargs)
        
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