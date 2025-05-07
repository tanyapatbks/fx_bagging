"""
Configuration parameters for the Forex Prediction System.
This module contains all the configurable parameters used in the project.
"""

import os
import numpy as np
import tensorflow as tf

# กำหนด seed เพื่อให้ผลลัพธ์เหมือนเดิมในทุกการรัน
np.random.seed(42)
tf.random.set_seed(42)

class Config:
    """
    Configuration parameters for the Forex Prediction System.
    """
    # พารามิเตอร์ทั่วไป
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    HYPERPARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hyperparams")
    
    # กำหนดช่วงเวลาสำหรับ training และ testing
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
    EPOCHS = 200
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    PATIENCE = 20  # จำนวนรอบที่ไม่มีการปรับปรุงก่อนหยุดการเทรน
    
        # LSTM parameters - ปรับค่าเริ่มต้นให้ดีขึ้น
    LSTM_PARAMS = {
        'learning_rate': 0.0005,
        'units_layer1': 128,
        'units_layer2': 64,
        'dropout1': 0.3,
        'dropout2': 0.3,
        'recurrent_dropout': 0.1,
        'l1_reg': 0.0001,
        'l2_reg': 0.0005
    }

    # GRU parameters - ปรับค่าเริ่มต้นให้ดีขึ้น
    GRU_PARAMS = {
        'learning_rate': 0.0005,
        'units_layer1': 160,
        'units_layer2': 80,
        'dropout1': 0.3,
        'dropout2': 0.3,
        'recurrent_dropout': 0.1,
        'l1_reg': 0.0001,
        'l2_reg': 0.0005
    }

    # XGBoost parameters - ปรับค่าเริ่มต้นให้ดีขึ้น
    XGB_PARAMS = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.005,
        'gamma': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'random_state': 42
    }

    # TFT parameters - ปรับค่าเริ่มต้นให้ดีขึ้น
    TFT_PARAMS = {
        'learning_rate': 0.0005,
        'attention_heads': 6,
        'dropout': 0.15,
        'hidden_units': 96,
        'hidden_continuous_size': 32
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

    MARKET_SPECIFIC_PARAMS = {
        'EURUSD': {
            'LSTM': {
                'learning_rate': 0.0003,
                'units_layer1': 160,
                'units_layer2': 80,
                'dropout1': 0.3,
                'dropout2': 0.2
            },
            'XGBoost': {
                'max_depth': 6,
                'min_child_weight': 4
            }
        },
        'GBPUSD': {
            'LSTM': {
                'learning_rate': 0.0005,
                'units_layer1': 192,
                'units_layer2': 64,
                'dropout1': 0.25,
                'dropout2': 0.25
            },
            'XGBoost': {
                'max_depth': 7,
                'min_child_weight': 3
            }
        },
        'USDJPY': {
            'LSTM': {
                'learning_rate': 0.0004,
                'units_layer1': 128,
                'units_layer2': 64,
                'dropout1': 0.35,
                'dropout2': 0.2
            },
            'XGBoost': {
                'max_depth': 5,
                'min_child_weight': 5
            }
        }
    }

    # Tuning specific configuration
    TUNING_CONFIG = {
        'min_trials': 30,           # Minimum number of trials per model
        'recommended_trials': 50,   # Recommended number of trials for better results
        'max_timeout': 14400,       # Maximum timeout in seconds (4 hours)
        'tune_objective': 'combined' # 'rmse', 'directional_accuracy', or 'combined'
    }
    
    # Feature Selection
    FEATURE_SELECTION_METHOD = 'random_forest'
    FEATURE_SELECTION_THRESHOLD = 'median'
    
    # Evaluation parameters
    INITIAL_BALANCE = 10000
    TRADING_COMMISSION = 0.0001  # 1 pip commission
    
    # สร้างโฟลเดอร์ที่จำเป็น
    @staticmethod
    def create_directories():
        """
        Create all necessary directories for the project
        """
        directories = [
            Config.MODELS_PATH, 
            Config.RESULTS_PATH, 
            Config.LOG_PATH,
            Config.HYPERPARAMS_PATH
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)