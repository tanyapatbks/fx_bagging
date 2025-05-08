"""
Configuration parameters for the Forex Prediction System.
This module contains all the configurable parameters used in the project.
"""

import os
import numpy as np
import tensorflow as tf
import json
from datetime import datetime, timedelta

# กำหนด seed เพื่อให้ผลลัพธ์เหมือนเดิมในทุกการรัน
np.random.seed(42)
tf.random.set_seed(42)

class Config:
    """
    Configuration parameters for the Forex Prediction System.
    """
    # โครงสร้างโปรเจค
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
    RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
    LOG_PATH = os.path.join(PROJECT_ROOT, "logs")
    HYPERPARAMS_PATH = os.path.join(PROJECT_ROOT, "hyperparams")
    REPORTS_PATH = os.path.join(PROJECT_ROOT, "reports")
    
    # กำหนดช่วงเวลาสำหรับ training และ testing
    # ตั้งค่าเริ่มต้น (สามารถเปลี่ยนแปลงได้ตามข้อมูลจริง)
    TRAIN_START = "2020-01-01"
    TRAIN_END = "2020-12-31"
    VALIDATION_START = "2021-01-01"
    VALIDATION_END = "2021-06-30"
    TEST_START = "2021-07-01"
    TEST_END = "2021-12-31"
    
    # คู่เงินที่ใช้ในการทดสอบ
    CURRENCY_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
    PAIR_SYMBOLS = {"EURUSD": "E", "GBPUSD": "G", "USDJPY": "J"}
    
    # พารามิเตอร์ทั่วไปสำหรับทุกโมเดล
    SEQUENCE_LENGTH = 24  # ใช้ข้อมูล 24 ชั่วโมงย้อนหลังในการทำนาย
    PREDICTION_HORIZON = 1  # ทำนายล่วงหน้า 1 ช่วงเวลา
    EPOCHS = 200
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    PATIENCE = 20  # จำนวนรอบที่ไม่มีการปรับปรุงก่อนหยุดการเทรน
    
    # ปรับ hyperparameters สำหรับ LSTM
    LSTM_PARAMS = {
        'learning_rate': 0.0003,
        'units_layer1': 192,
        'units_layer2': 96,
        'dropout1': 0.25,
        'dropout2': 0.25,
        'recurrent_dropout': 0.1,
        'l1_reg': 0.00005,
        'l2_reg': 0.0002,
        'attention_units': 64,
        'use_bidirectional': True,
        'use_layer_norm': True
    }

    # สำหรับ GRU
    GRU_PARAMS = {
        'learning_rate': 0.0003,
        'units_layer1': 224,
        'units_layer2': 112,
        'dropout1': 0.2,
        'dropout2': 0.2,
        'recurrent_dropout': 0.05,
        'l1_reg': 0.00005,
        'l2_reg': 0.0001,
        'attention_units': 64,
        'use_bidirectional': True,
        'use_layer_norm': True,
        'use_attention': True
    }
    
    # ปรับ hyperparameters สำหรับ XGBoost
    XGB_PARAMS = {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.003,
        'gamma': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.2,
        'reg_lambda': 1.2,
        'random_state': 42,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'n_jobs': -1
    }

    # ปรับ hyperparameters สำหรับ TFT
    TFT_PARAMS = {
        'learning_rate': 0.0003,
        'attention_heads': 8,
        'dropout': 0.15,
        'hidden_units': 128,
        'hidden_continuous_size': 48
    }
    
    # Feature Engineering parameters
    TECHNICAL_INDICATORS = {
        'sma': [5, 10, 20, 50],
        'ema': [12, 26],
        'macd': True,
        'rsi': [14],
        'bollinger': [20],
        'stochastic': [14, 3],
        'atr': [14],
        'adx': [14],
        'momentum': [10],
        'roc': [10]
    }

    # พารามิเตอร์เฉพาะสำหรับแต่ละคู่สกุลเงิน
    PAIR_SPECIFIC_PARAMS = {
        'EURUSD': {
            'LSTM': {
                'l1_reg': 0.0002,
                'attention_units': 72
            },
            'GRU': {
                'units_layer1': 256,
                'recurrent_dropout': 0.08
            },
            'XGBoost': {
                'max_depth': 6,
                'min_child_weight': 4,
                'subsample': 0.85
            }
        },
        'GBPUSD': {
            'LSTM': {
                'dropout1': 0.3,
                'learning_rate': 0.0002
            },
            'GRU': {
                'learning_rate': 0.0004,
                'dropout1': 0.25
            },
            'XGBoost': {
                'max_depth': 7,
                'min_child_weight': 3,
                'subsample': 0.75
            }
        },
        'USDJPY': {
            'LSTM': {
                'units_layer1': 224,
                'recurrent_dropout': 0.15
            },
            'GRU': {
                'units_layer1': 256,
                'dropout1': 0.15
            },
            'XGBoost': {
                'max_depth': 5,
                'min_child_weight': 5,
                'gamma': 0.1
            }
        }
    }

    # Tuning specific configuration
    TUNING_CONFIG = {
        'min_trials': 30,           # Minimum number of trials per model
        'recommended_trials': 50,   # Recommended number of trials for better results
        'max_timeout': 14400,       # Maximum timeout in seconds (4 hours)
        'tune_objective': 'combined', # 'rmse', 'directional_accuracy', or 'combined'
        'pruner': 'median',          # 'median', 'percentile', or 'none'
        'sampler': 'tpe'             # 'tpe', 'random', or 'grid'
    }
    
    # Feature Selection
    FEATURE_SELECTION_METHOD = 'random_forest'  # 'random_forest', 'select_from_model', 'rfe'
    FEATURE_SELECTION_THRESHOLD = 'median'  # 'median', 'mean', หรือค่าจริง (เช่น 0.01)
    FEATURE_SELECTION_NFEATURES = 0.5  # สัดส่วนของคุณลักษณะที่ต้องการเลือก (0.0-1.0)
    
    # Evaluation parameters
    INITIAL_BALANCE = 10000  # เงินต้นสำหรับการซื้อขาย
    TRADING_COMMISSION = 0.0001  # ค่าธรรมเนียมการซื้อขาย (1 pip)
    RISK_PER_TRADE = 0.02  # สัดส่วนของเงินต้นที่ยอมเสี่ยงต่อการซื้อขาย (2%)
    STOP_LOSS_PCT = 0.01  # เปอร์เซ็นต์การหยุดขาดทุน (1%)
    TAKE_PROFIT_PCT = 0.02  # เปอร์เซ็นต์การทำกำไร (2%)
    
    # บาคยะ (Bagging) และการรวมโมเดล
    ENSEMBLE_WEIGHTS = {
        'LSTM': 0.3,
        'GRU': 0.3,
        'XGBoost': 0.3,
        'TFT': 0.1
    }
    USE_WEIGHTED_ENSEMBLE = True
    OPTIMIZE_ENSEMBLE_WEIGHTS = True
    
    # พารามิเตอร์เฉพาะสำหรับการเทรดทดลอง
    TRADING_PARAMS = {
        'use_stop_loss': True,
        'use_take_profit': True,
        'trailing_stop': True,
        'trailing_stop_activation': 0.005,  # เปิดใช้ trailing stop เมื่อกำไร 0.5%
        'max_positions': 3,  # จำนวนสูงสุดของสถานะที่เปิดพร้อมกัน
        'position_sizing': 'fixed_risk',  # 'fixed_risk', 'fixed_amount', 'percentage'
        'leverage': 1.0  # คูณด้วย 1 เท่า (ไม่ใช้ leverage)
    }
    
    # สร้างโฟลเดอร์ที่จำเป็น
    @classmethod
    def create_directories(cls):
        """
        Create all necessary directories for the project
        """
        directories = [
            cls.DATA_PATH, 
            cls.MODELS_PATH, 
            cls.RESULTS_PATH, 
            cls.LOG_PATH,
            cls.HYPERPARAMS_PATH,
            cls.REPORTS_PATH
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    @classmethod
    def save_config(cls, filename="config_backup.json"):
        """
        Save configuration to a JSON file
        
        Args:
            filename: Name of the JSON file to save
        """
        # รวบรวมค่า config ที่ต้องการบันทึก
        config_dict = {
            'TRAIN_START': cls.TRAIN_START,
            'TRAIN_END': cls.TRAIN_END,
            'VALIDATION_START': cls.VALIDATION_START,
            'VALIDATION_END': cls.VALIDATION_END,
            'TEST_START': cls.TEST_START,
            'TEST_END': cls.TEST_END,
            'CURRENCY_PAIRS': cls.CURRENCY_PAIRS,
            'PAIR_SYMBOLS': cls.PAIR_SYMBOLS,
            'SEQUENCE_LENGTH': cls.SEQUENCE_LENGTH,
            'PREDICTION_HORIZON': cls.PREDICTION_HORIZON,
            'EPOCHS': cls.EPOCHS,
            'BATCH_SIZE': cls.BATCH_SIZE,
            'VALIDATION_SPLIT': cls.VALIDATION_SPLIT,
            'PATIENCE': cls.PATIENCE,
            'LSTM_PARAMS': cls.LSTM_PARAMS,
            'GRU_PARAMS': cls.GRU_PARAMS,
            'XGB_PARAMS': cls.XGB_PARAMS,
            'TFT_PARAMS': cls.TFT_PARAMS,
            'TECHNICAL_INDICATORS': cls.TECHNICAL_INDICATORS,
            'PAIR_SPECIFIC_PARAMS': cls.PAIR_SPECIFIC_PARAMS,
            'TUNING_CONFIG': cls.TUNING_CONFIG,
            'FEATURE_SELECTION_METHOD': cls.FEATURE_SELECTION_METHOD,
            'FEATURE_SELECTION_THRESHOLD': cls.FEATURE_SELECTION_THRESHOLD,
            'FEATURE_SELECTION_NFEATURES': cls.FEATURE_SELECTION_NFEATURES,
            'INITIAL_BALANCE': cls.INITIAL_BALANCE,
            'TRADING_COMMISSION': cls.TRADING_COMMISSION,
            'RISK_PER_TRADE': cls.RISK_PER_TRADE,
            'STOP_LOSS_PCT': cls.STOP_LOSS_PCT,
            'TAKE_PROFIT_PCT': cls.TAKE_PROFIT_PCT,
            'ENSEMBLE_WEIGHTS': cls.ENSEMBLE_WEIGHTS,
            'USE_WEIGHTED_ENSEMBLE': cls.USE_WEIGHTED_ENSEMBLE,
            'OPTIMIZE_ENSEMBLE_WEIGHTS': cls.OPTIMIZE_ENSEMBLE_WEIGHTS,
            'TRADING_PARAMS': cls.TRADING_PARAMS
        }
        
        # สร้างโฟลเดอร์ถ้ายังไม่มี
        cls.create_directories()
        
        # บันทึกไฟล์
        filepath = os.path.join(cls.LOG_PATH, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
            print(f"Configuration saved to {filepath}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    @classmethod
    def load_config(cls, filename="config_backup.json"):
        """
        Load configuration from a JSON file
        
        Args:
            filename: Name of the JSON file to load
        
        Returns:
            True if loaded successfully, False otherwise
        """
        # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
        filepath = os.path.join(cls.LOG_PATH, filename)
        
        if not os.path.exists(filepath):
            print(f"Configuration file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # อัปเดตค่า config
            for key, value in config_dict.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
            
            print(f"Configuration loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    @classmethod
    def update_train_test_dates(cls, train_years=1, test_months=6, validation_months=6):
        """
        Update training, validation, and testing date ranges based on current date
        
        Args:
            train_years: Number of years for training data
            test_months: Number of months for testing data
            validation_months: Number of months for validation data
        """
        # ใช้วันที่ปัจจุบัน
        today = datetime.now()
        
        # กำหนดช่วงทดสอบ
        test_end = today.strftime("%Y-%m-%d")
        test_start = (today - timedelta(days=30*test_months)).strftime("%Y-%m-%d")
        
        # กำหนดช่วงตรวจสอบ
        validation_end = (today - timedelta(days=30*test_months)).strftime("%Y-%m-%d")
        validation_start = (today - timedelta(days=30*(test_months+validation_months))).strftime("%Y-%m-%d")
        
        # กำหนดช่วงฝึกฝน
        train_end = (today - timedelta(days=30*(test_months+validation_months))).strftime("%Y-%m-%d")
        train_start = (today - timedelta(days=365*train_years + 30*(test_months+validation_months))).strftime("%Y-%m-%d")
        
        # อัปเดตค่า config
        cls.TRAIN_START = train_start
        cls.TRAIN_END = train_end
        cls.VALIDATION_START = validation_start
        cls.VALIDATION_END = validation_end
        cls.TEST_START = test_start
        cls.TEST_END = test_end
        
        print(f"Updated date ranges:")
        print(f"Training: {train_start} to {train_end}")
        print(f"Validation: {validation_start} to {validation_end}")
        print(f"Testing: {test_start} to {test_end}")
    
    @classmethod
    def get_model_params(cls, model_type, pair=None):
        """
        Get model parameters with pair-specific overrides if available
        
        Args:
            model_type: Type of model ('LSTM', 'GRU', 'XGBoost', 'TFT')
            pair: Currency pair (optional)
            
        Returns:
            Dictionary of model parameters
        """
        # รับพารามิเตอร์พื้นฐาน
        if model_type == 'LSTM':
            params = cls.LSTM_PARAMS.copy()
        elif model_type == 'GRU':
            params = cls.GRU_PARAMS.copy()
        elif model_type == 'XGBoost' or model_type == 'XGB':
            params = cls.XGB_PARAMS.copy()
        elif model_type == 'TFT':
            params = cls.TFT_PARAMS.copy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # ถ้ามี pair และมีพารามิเตอร์เฉพาะสำหรับคู่สกุลเงินนั้น
        if pair is not None and pair in cls.PAIR_SPECIFIC_PARAMS:
            pair_params = cls.PAIR_SPECIFIC_PARAMS[pair]
            if model_type in pair_params:
                # อัปเดตพารามิเตอร์ด้วยค่าเฉพาะสำหรับคู่สกุลเงิน
                params.update(pair_params[model_type])
        
        return params