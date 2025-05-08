"""
LSTM Model for Forex Prediction
This module contains the implementation of the LSTM model.
"""

import os
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, 
    Input, Multiply, Lambda,
    Bidirectional, TimeDistributed,
    LayerNormalization, Concatenate,
    Attention, Reshape
)
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
        
        # ตรวจสอบว่า TensorFlow GPU พร้อมใช้งานหรือไม่
        self._check_gpu()
    
    def _check_gpu(self):
        """
        Check if TensorFlow can see GPU and print relevant information
        """
        try:
            # ตรวจสอบอุปกรณ์ที่ใช้ได้
            physical_devices = tf.config.list_physical_devices()
            gpu_devices = tf.config.list_physical_devices('GPU')
            
            if gpu_devices:
                print(f"TensorFlow is using {len(gpu_devices)} GPU(s):")
                for gpu in gpu_devices:
                    print(f"  - {gpu.name}")
                
                # ตั้งค่าการจัดสรรหน่วยความจำ
                for gpu in gpu_devices:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"  - Memory growth enabled for {gpu.name}")
                    except:
                        print(f"  - Could not enable memory growth for {gpu.name}")
            else:
                print("No GPU found. TensorFlow will use CPU.")
        except Exception as e:
            print(f"Error checking GPU: {e}")
    
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
        use_bidirectional = self.params.get('use_bidirectional', False)
        use_layer_norm = self.params.get('use_layer_norm', False)
        
        # แสดงข้อมูลการสร้างโมเดล
        print("\n--- Building LSTM Model ---")
        print(f"Input Shape: {input_shape}")
        print(f"Layer 1 Units: {units_layer1}")
        print(f"Layer 2 Units: {units_layer2}")
        print(f"Attention Units: {attention_units}")
        print(f"Dropouts: {dropout1}, {dropout2}, Recurrent: {recurrent_dropout}")
        print(f"Regularization: L1={l1_reg}, L2={l2_reg}")
        print(f"Bidirectional: {use_bidirectional}")
        print(f"Layer Normalization: {use_layer_norm}")
        
        # สร้างโมเดล
        inputs = Input(shape=input_shape, name='input_layer')
        
        # Layer 1: LSTM with optional bidirectional wrapper
        if use_bidirectional:
            lstm1 = Bidirectional(
                LSTM(
                    units_layer1, 
                    return_sequences=True, 
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    name='lstm_layer1'
                ),
                name='bidirectional_lstm1'
            )(inputs)
        else:
            lstm1 = LSTM(
                units_layer1, 
                return_sequences=True, 
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name='lstm_layer1'
            )(inputs)
        
        # Optional Layer Normalization
        if use_layer_norm:
            lstm1 = LayerNormalization(epsilon=1e-6, name='layer_norm1')(lstm1)
        
        lstm1 = Dropout(dropout1, name='dropout_layer1')(lstm1)
        
        # Attention mechanism
        attention = Dense(attention_units, activation='tanh', name='attention_dense')(lstm1)
        attention = Dense(1, activation='softmax', name='attention_weights')(attention)
        context_vector = Multiply(name='attention_multiply')([lstm1, attention])
        context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1), name='attention_sum')(context_vector)
        
        # Layer 2: Optional second LSTM layer
        if units_layer2 > 0:
            # Reshape for the second LSTM
            reshaped_context = Reshape((1, -1))(context_vector)
            
            if use_bidirectional:
                lstm2 = Bidirectional(
                    LSTM(
                        units_layer2, 
                        recurrent_dropout=recurrent_dropout,
                        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                        name='lstm_layer2'
                    ),
                    name='bidirectional_lstm2'
                )(reshaped_context)
            else:
                lstm2 = LSTM(
                    units_layer2, 
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    name='lstm_layer2'
                )(reshaped_context)
            
            if use_layer_norm:
                lstm2 = LayerNormalization(epsilon=1e-6, name='layer_norm2')(lstm2)
                
            lstm2 = Dropout(dropout2, name='dropout_layer2')(lstm2)
            outputs = Dense(1, name='output_layer')(lstm2)
        else:
            # ถ้าไม่ใช้ LSTM Layer ที่สอง
            outputs = Dense(1, name='output_layer')(context_vector)
        
        # สร้างโมเดล
        model = Model(inputs=inputs, outputs=outputs)
        
        # คอมไพล์โมเดล
        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.0005))
        model.compile(
            optimizer=optimizer, 
            loss='mse',
            metrics=[RootMeanSquaredError()]
        )
        
        # Print model summary
        model.summary()
        
        return model
    
    def build_stacked_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        สร้างโมเดล LSTM แบบซ้อนหลายชั้น
        
        Args:
            input_shape: รูปร่างของข้อมูลนำเข้า (sequence_length, num_features)
            
        Returns:
            โมเดล LSTM ที่คอมไพล์แล้ว
        """
        # อ่านพารามิเตอร์
        num_layers = self.params.get('num_layers', 2)
        units = self.params.get('units', [128, 64])
        dropout_rates = self.params.get('dropout_rates', [0.3, 0.3])
        recurrent_dropout = self.params.get('recurrent_dropout', 0.1)
        l1_reg = self.params.get('l1_reg', 0.0001)
        l2_reg = self.params.get('l2_reg', 0.0005)
        
        # ตรวจสอบพารามิเตอร์
        if isinstance(units, int):
            units = [units] * num_layers
        if isinstance(dropout_rates, float):
            dropout_rates = [dropout_rates] * num_layers
        
        # ปรับให้มีความยาวเท่ากับจำนวนชั้น
        units = units[:num_layers] + [units[-1]] * (num_layers - len(units))
        dropout_rates = dropout_rates[:num_layers] + [dropout_rates[-1]] * (num_layers - len(dropout_rates))
        
        # แสดงข้อมูลการสร้างโมเดล
        print("\n--- Building Stacked LSTM Model ---")
        print(f"Input Shape: {input_shape}")
        print(f"Number of Layers: {num_layers}")
        print(f"Units per Layer: {units}")
        print(f"Dropout Rates: {dropout_rates}")
        print(f"Recurrent Dropout: {recurrent_dropout}")
        print(f"Regularization: L1={l1_reg}, L2={l2_reg}")
        
        # สร้างโมเดล
        model = Sequential()
        
        # ชั้นแรก
        model.add(LSTM(
            units[0],
            input_shape=input_shape,
            return_sequences=(num_layers > 1),  # return sequences only if more layers
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
            name=f'lstm_layer1'
        ))
        model.add(Dropout(dropout_rates[0], name=f'dropout_layer1'))
        
        # ชั้นกลาง
        for i in range(1, num_layers - 1):
            model.add(LSTM(
                units[i],
                return_sequences=True,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'lstm_layer{i+1}'
            ))
            model.add(Dropout(dropout_rates[i], name=f'dropout_layer{i+1}'))
        
        # ชั้นสุดท้าย
        if num_layers > 1:
            model.add(LSTM(
                units[-1],
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'lstm_layer{num_layers}'
            ))
            model.add(Dropout(dropout_rates[-1], name=f'dropout_layer{num_layers}'))
        
        # ชั้น output
        model.add(Dense(1, name='output_layer'))
        
        # คอมไพล์โมเดล
        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.0005))
        model.compile(
            optimizer=optimizer, 
            loss='mse',
            metrics=[RootMeanSquaredError()]
        )
        
        # Print model summary
        model.summary()
        
        return model
    
    def build_multi_horizon_attention(self, inputs, sequence_length, feature_dim):
        """
        สร้าง Attention สำหรับหลายช่วงเวลา
        
        Args:
            inputs: ข้อมูลนำเข้า
            sequence_length: ความยาวของลำดับ
            feature_dim: มิติของคุณลักษณะ
            
        Returns:
            Context vector
        """
        # Attention ระยะสั้น (ไม่กี่ชั่วโมงที่ผ่านมา)
        short_window = min(6, sequence_length)
        attention_short = Dense(32, activation='tanh')(inputs[:, -short_window:, :])
        attention_short = Dense(1, activation='softmax')(attention_short)
        context_short = Multiply()([inputs[:, -short_window:, :], attention_short])
        context_short = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_short)
        
        # Attention ระยะกลาง (กลางลำดับ)
        mid_start = max(0, sequence_length - 24)
        mid_end = sequence_length - short_window
        
        if mid_end > mid_start:
            attention_mid = Dense(32, activation='tanh')(inputs[:, mid_start:mid_end, :])
            attention_mid = Dense(1, activation='softmax')(attention_mid)
            context_mid = Multiply()([inputs[:, mid_start:mid_end, :], attention_mid])
            context_mid = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_mid)
        else:
            # ถ้าไม่มีช่วงกลาง ให้ใช้ค่าศูนย์
            context_mid = Lambda(lambda x: tf.zeros_like(x[:, 0, :]))(inputs)
        
        # Attention ระยะยาว (ลำดับทั้งหมด)
        attention_long = Dense(32, activation='tanh')(inputs)
        attention_long = Dense(1, activation='softmax')(attention_long)
        context_long = Multiply()([inputs, attention_long])
        context_long = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_long)
        
        # รวมบริบท
        combined_context = Concatenate()([context_short, context_mid, context_long])
        return combined_context

    def train(self, X_train: np.ndarray, y_train: np.ndarray, model_name: str, 
          callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
          epochs: Optional[int] = None,
          validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
          model_type: str = 'attention') -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features (shape: [samples, sequence_length, features])
            y_train: Training target values
            model_name: Name for saving the model
            callbacks: Optional list of Keras callbacks
            epochs: Optional number of epochs (default: use config.EPOCHS)
            validation_data: Optional tuple of (X_val, y_val) for validation
            model_type: Type of model to build ('attention' or 'stacked')
                
        Returns:
            Tuple of (trained_model, training_history)
        """
        # ตรวจสอบข้อมูลนำเข้า
        if len(X_train.shape) != 3:
            raise ValueError(f"Expected X_train shape of [samples, sequence_length, features], got {X_train.shape}")
            
        # สร้างโมเดล
        if model_type == 'stacked':
            model = self.build_stacked_lstm_model(X_train.shape[1:])
        else:
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
        
        # บันทึกโมเดล (ถ้ายังไม่ได้บันทึกโดย ModelCheckpoint)
        try:
            if not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
                model.save(os.path.join(self.config.MODELS_PATH, f"{model_name}.h5"))
                print(f"Model saved as {model_name}.h5")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
        
        return model, history
    
    def predict(self, model: tf.keras.Model, X_test: np.ndarray, 
                scaler: Any = None, target_idx: int = -1) -> np.ndarray:
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
        # ตรวจสอบข้อมูลนำเข้า
        if X_test is None or len(X_test) == 0:
            return np.array([])
        
        if len(X_test.shape) != 3:
            raise ValueError(f"Expected X_test shape of [samples, sequence_length, features], got {X_test.shape}")
        
        # ทำนาย
        y_pred_scaled = model.predict(X_test)
        
        # แปลงกลับสเกลเดิม ถ้ามี scaler
        if scaler is not None:
            # ตรวจสอบว่า target_idx ถูกต้องหรือไม่
            if not hasattr(scaler, 'scale_') or target_idx >= len(scaler.scale_) or target_idx < 0:
                raise ValueError(f"Invalid target_idx {target_idx} for scaler with shape {scaler.scale_.shape if hasattr(scaler, 'scale_') else 'unknown'}")
                
            # สร้าง array สำหรับการแปลงกลับ
            dummy = np.zeros((len(y_pred_scaled), scaler.scale_.shape[0]))
            dummy[:, target_idx] = y_pred_scaled.flatten()
            
            # แปลงกลับเฉพาะคอลัมน์เป้าหมาย
            y_pred = scaler.inverse_transform(dummy)[:, target_idx]
        else:
            # ถ้าไม่มี scaler ให้ใช้ค่าทำนายโดยตรง
            y_pred = y_pred_scaled.flatten()
        
        return y_pred
    
    def load_model(self, model_path: str) -> tf.keras.Model:
        """
        Load a trained model from file
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")
    
    def create_sequence_aware_batches(self, X: np.ndarray, y: np.ndarray, target_idx: int, 
                                  batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create batches based on sequence pattern similarity
        
        Args:
            X: Input sequences
            y: Target values
            target_idx: Index of target column in sequences
            batch_size: Batch size
            
        Returns:
            Tuple of sorted (X, y)
        """
        if len(X) <= 1:
            return X, y
            
        # คำนวณลายเซ็นรูปแบบอย่างง่ายสำหรับแต่ละลำดับ (การเปลี่ยนทิศทาง)
        signatures = []
        for seq in X:
            # นับการเปลี่ยนทิศทางในราคาปิด
            closes = seq[:, target_idx]
            directions = np.sign(np.diff(closes))
            direction_changes = np.sum(np.abs(np.diff(directions)))
            signatures.append(direction_changes)
        
        # เรียงตามลายเซ็น
        sort_idx = np.argsort(signatures)
        return X[sort_idx], y[sort_idx]