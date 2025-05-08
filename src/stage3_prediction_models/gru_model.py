"""
GRU Model for Forex Prediction
This module contains the implementation of the Gated Recurrent Unit (GRU) model.
"""

import os
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, GRU, Dropout, 
    Input, Multiply, Lambda,
    Bidirectional, TimeDistributed,
    LayerNormalization, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.regularizers import l1_l2


class GRUModel:
    """
    Gated Recurrent Unit Neural Network model for time series prediction
    """
    
    def __init__(self, config, params=None):
        """
        Initialize the GRU model with configuration parameters
        
        Args:
            config: Configuration object containing model parameters
            params: Optional dictionary of model hyperparameters
        """
        self.config = config
        self.params = params if params is not None else config.GRU_PARAMS
        
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
        Build and compile the GRU model architecture
        
        Args:
            input_shape: Shape of the input data (sequence_length, num_features)
            
        Returns:
            Compiled GRU model
        """
        # อ่านพารามิเตอร์
        units_layer1 = self.params.get('units_layer1', 128)
        units_layer2 = self.params.get('units_layer2', 64)
        dropout1 = self.params.get('dropout1', 0.3)
        dropout2 = self.params.get('dropout2', 0.3)
        recurrent_dropout = self.params.get('recurrent_dropout', 0.2)
        l1_reg = self.params.get('l1_reg', 0.0001)
        l2_reg = self.params.get('l2_reg', 0.0001)
        use_bidirectional = self.params.get('use_bidirectional', False)
        use_layer_norm = self.params.get('use_layer_norm', False)
        use_attention = self.params.get('use_attention', True)
        attention_units = self.params.get('attention_units', 32)
        
        # แสดงข้อมูลการสร้างโมเดล
        print("\n--- Building GRU Model ---")
        print(f"Input Shape: {input_shape}")
        print(f"Layer 1 Units: {units_layer1}")
        print(f"Layer 2 Units: {units_layer2}")
        print(f"Dropouts: {dropout1}, {dropout2}, Recurrent: {recurrent_dropout}")
        print(f"Regularization: L1={l1_reg}, L2={l2_reg}")
        print(f"Bidirectional: {use_bidirectional}")
        print(f"Layer Normalization: {use_layer_norm}")
        print(f"Attention: {use_attention}, Units: {attention_units}")
        
        # สร้างโมเดลโดยใช้ Functional API
        inputs = Input(shape=input_shape, name='input_layer')
        
        # GRU layer 1
        if use_bidirectional:
            gru1 = Bidirectional(
                GRU(
                    units_layer1, 
                    return_sequences=True, 
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    name='gru_layer1'
                ),
                name='bidirectional_gru1'
            )(inputs)
        else:
            gru1 = GRU(
                units_layer1, 
                return_sequences=True, 
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name='gru_layer1'
            )(inputs)
        
        if use_layer_norm:
            gru1 = LayerNormalization(epsilon=1e-6, name='layer_norm1')(gru1)
            
        gru1 = Dropout(dropout1, name='dropout_layer1')(gru1)
        
        # Attention mechanism (optional)
        if use_attention:
            # Self-attention
            attention = Dense(attention_units, activation='tanh', name='attention_dense')(gru1)
            attention = Dense(1, activation='softmax', name='attention_weights')(attention)
            context_vector = Multiply(name='attention_multiply')([gru1, attention])
            context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1), name='attention_sum')(context_vector)
        else:
            # No attention - GRU layer 2 will process all timesteps
            context_vector = gru1
        
        # GRU layer 2
        if units_layer2 > 0:
            if use_attention:
                # Reshape context vector for GRU input
                reshaped_context = Lambda(lambda x: tf.expand_dims(x, axis=1))(context_vector)
                
                if use_bidirectional:
                    gru2 = Bidirectional(
                        GRU(
                            units_layer2, 
                            recurrent_dropout=recurrent_dropout,
                            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                            name='gru_layer2'
                        ),
                        name='bidirectional_gru2'
                    )(reshaped_context)
                else:
                    gru2 = GRU(
                        units_layer2, 
                        recurrent_dropout=recurrent_dropout,
                        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                        name='gru_layer2'
                    )(reshaped_context)
            else:
                # If not using attention, use the output from first GRU layer
                if use_bidirectional:
                    gru2 = Bidirectional(
                        GRU(
                            units_layer2, 
                            recurrent_dropout=recurrent_dropout,
                            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                            name='gru_layer2'
                        ),
                        name='bidirectional_gru2'
                    )(gru1)
                else:
                    gru2 = GRU(
                        units_layer2, 
                        recurrent_dropout=recurrent_dropout,
                        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                        name='gru_layer2'
                    )(gru1)
            
            if use_layer_norm:
                gru2 = LayerNormalization(epsilon=1e-6, name='layer_norm2')(gru2)
                
            gru2 = Dropout(dropout2, name='dropout_layer2')(gru2)
            outputs = Dense(1, name='output_layer')(gru2)
        else:
            # If no second GRU layer, output directly from first layer or attention
            if use_attention:
                outputs = Dense(1, name='output_layer')(context_vector)
            else:
                # For single GRU layer without attention, take only the last output
                x = Lambda(lambda x: x[:, -1, :])(gru1)
                outputs = Dense(1, name='output_layer')(x)
        
        # สร้างโมเดล
        model = Model(inputs=inputs, outputs=outputs)
        
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
    
    def build_stacked_gru_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build a stacked GRU model with multiple layers
        
        Args:
            input_shape: Shape of the input data (sequence_length, num_features)
            
        Returns:
            Compiled stacked GRU model
        """
        # อ่านพารามิเตอร์
        num_layers = self.params.get('num_layers', 2)
        units = self.params.get('units', [128, 64])
        dropout_rates = self.params.get('dropout_rates', [0.3, 0.3])
        recurrent_dropout = self.params.get('recurrent_dropout', 0.2)
        l1_reg = self.params.get('l1_reg', 0.0001)
        l2_reg = self.params.get('l2_reg', 0.0001)
        
        # ตรวจสอบพารามิเตอร์
        if isinstance(units, int):
            units = [units] * num_layers
        if isinstance(dropout_rates, float):
            dropout_rates = [dropout_rates] * num_layers
        
        # ปรับให้มีความยาวเท่ากับจำนวนชั้น
        units = units[:num_layers] + [units[-1]] * (num_layers - len(units))
        dropout_rates = dropout_rates[:num_layers] + [dropout_rates[-1]] * (num_layers - len(dropout_rates))
        
        # แสดงข้อมูลการสร้างโมเดล
        print("\n--- Building Stacked GRU Model ---")
        print(f"Input Shape: {input_shape}")
        print(f"Number of Layers: {num_layers}")
        print(f"Units per Layer: {units}")
        print(f"Dropout Rates: {dropout_rates}")
        print(f"Recurrent Dropout: {recurrent_dropout}")
        print(f"Regularization: L1={l1_reg}, L2={l2_reg}")
        
        # สร้างโมเดล
        model = Sequential()
        
        # ชั้นแรก
        model.add(GRU(
            units[0],
            input_shape=input_shape,
            return_sequences=(num_layers > 1),  # return sequences only if more layers
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
            name=f'gru_layer1'
        ))
        model.add(Dropout(dropout_rates[0], name=f'dropout_layer1'))
        
        # ชั้นกลาง
        for i in range(1, num_layers - 1):
            model.add(GRU(
                units[i],
                return_sequences=True,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'gru_layer{i+1}'
            ))
            model.add(Dropout(dropout_rates[i], name=f'dropout_layer{i+1}'))
        
        # ชั้นสุดท้าย
        if num_layers > 1:
            model.add(GRU(
                units[-1],
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'gru_layer{num_layers}'
            ))
            model.add(Dropout(dropout_rates[-1], name=f'dropout_layer{num_layers}'))
        
        # ชั้น output
        model.add(Dense(1, name='output_layer'))
        
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
          callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
          epochs: Optional[int] = None,
          validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
          model_type: str = 'default') -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """
        Train the GRU model
        
        Args:
            X_train: Training features (shape: [samples, sequence_length, features])
            y_train: Training target values
            model_name: Name for saving the model
            callbacks: Optional list of Keras callbacks
            epochs: Optional number of epochs (default: use config.EPOCHS)
            validation_data: Optional tuple of (X_val, y_val) for validation
            model_type: Type of model to build ('default' or 'stacked')
                
        Returns:
            Tuple of (trained_model, training_history)
        """
        # ตรวจสอบข้อมูลนำเข้า
        if len(X_train.shape) != 3:
            raise ValueError(f"Expected X_train shape of [samples, sequence_length, features], got {X_train.shape}")
        
        # สร้างโมเดล
        if model_type == 'stacked':
            model = self.build_stacked_gru_model(X_train.shape[1:])
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
            model: Trained GRU model
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
    
    def export_model_to_onnx(self, model: tf.keras.Model, output_path: str, input_shape: Tuple[int, int, int]):
        """
        Export model to ONNX format for deployment
        
        Args:
            model: Trained GRU model
            output_path: Path to save the ONNX model
            input_shape: Input shape as (batch_size, sequence_length, num_features)
        """
        try:
            import tf2onnx
            import onnx
            
            # Convert the model
            onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec(input_shape, tf.float32)])
            
            # Save the model
            onnx.save(onnx_model, output_path)
            print(f"Model exported to ONNX format: {output_path}")
        except ImportError:
            print("Error: tf2onnx or onnx package not installed. Run 'pip install tf2onnx onnx'")
        except Exception as e:
            print(f"Error exporting model to ONNX: {e}")
    
    def calculate_model_complexity(self, model: tf.keras.Model) -> Dict[str, int]:
        """
        Calculate model complexity metrics
        
        Args:
            model: Keras model
            
        Returns:
            Dictionary with complexity metrics
        """
        # จำนวนพารามิเตอร์ทั้งหมด
        total_params = model.count_params()
        
        # จำนวนพารามิเตอร์ที่ฝึกฝนได้
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        # จำนวนพารามิเตอร์ที่ไม่ฝึกฝน
        non_trainable_params = total_params - trainable_params
        
        # จำนวนชั้น
        num_layers = len(model.layers)
        
        # จำนวนชั้น GRU
        gru_layers = [layer for layer in model.layers if 'gru' in layer.name.lower()]
        num_gru_layers = len(gru_layers)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'num_layers': num_layers,
            'num_gru_layers': num_gru_layers
        }