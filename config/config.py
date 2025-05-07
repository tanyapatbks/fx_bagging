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
    
    # ปรับ hyperparameters สำหรับ LSTM/GRU
    LSTM_PARAMS = {
        'learning_rate': 0.0003,  # ลดลงเพื่อให้การเรียนรู้เสถียรมากขึ้น
        'units_layer1': 192,      # เพิ่มขึ้นเพื่อจับรูปแบบที่ซับซ้อนมากขึ้น
        'units_layer2': 96,       # เพิ่มขึ้นเพื่อประมวลผลข้อมูลที่ซับซ้อนมากขึ้น
        'dropout1': 0.25,         # ปรับลงเล็กน้อยเพื่อให้โมเดลเรียนรู้รูปแบบได้ดีขึ้น
        'dropout2': 0.25,
        'recurrent_dropout': 0.1,
        'l1_reg': 0.00005,        # ลดลงเพื่อให้โมเดลสามารถจับรูปแบบละเอียดได้ดีขึ้น
        'l2_reg': 0.0002,         # ลดลงเพื่อลดการ regularize ที่อาจมากเกินไป
        'attention_units': 64     # ใช้กับโมเดลที่มี attention
    }

    # สำหรับ GRU
    GRU_PARAMS = {
        'learning_rate': 0.0003,
        'units_layer1': 224,      # GRU อาจต้องการหน่วยมากกว่า LSTM เล็กน้อย
        'units_layer2': 112,
        'dropout1': 0.2,          # GRU มีประสิทธิภาพดีกับ dropout น้อยกว่า LSTM
        'dropout2': 0.2,
        'recurrent_dropout': 0.05,
        'l1_reg': 0.00005,
        'l2_reg': 0.0001,
        'attention_units': 64
    }
    # ปรับ hyperparameters สำหรับ XGBoost
    XGB_PARAMS = {
        'n_estimators': 500,      # เพิ่มขึ้นเพื่อให้มีต้นไม้มากขึ้น
        'max_depth': 5,           # ลดลงเพื่อป้องกัน overfitting
        'learning_rate': 0.003,   # ลดลงเพื่อให้การเรียนรู้เสถียรมากขึ้น
        'gamma': 0.05,            # ลดลงเพื่อให้สร้างโหนดใหม่ได้ง่ายขึ้น
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,    # เพิ่มขึ้นเพื่อป้องกัน overfitting
        'reg_alpha': 0.2,         # L1 regularization
        'reg_lambda': 1.2,        # L2 regularization
        'random_state': 42,
        'objective': 'reg:squarederror',
        'tree_method': 'hist'
    }

    # ปรับ hyperparameters สำหรับ TFT
    TFT_PARAMS = {
        'learning_rate': 0.0003,
        'attention_heads': 8,     # เพิ่มขึ้นเพื่อจับความสัมพันธ์ที่ซับซ้อนมากขึ้น
        'dropout': 0.15,
        'hidden_units': 128,      # เพิ่มขึ้นเพื่อเพิ่มความสามารถในการเรียนรู้
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

    # ในไฟล์ config/config.py ภายใน Config class
    
    # พารามิเตอร์เฉพาะสำหรับแต่ละคู่สกุลเงิน
    PAIR_SPECIFIC_PARAMS = {
        'EURUSD': {
            'LSTM': {
                'l1_reg': 0.0002,
                'attention_units': 72
            }
        },
        'GBPUSD': {
            'LSTM': {
                'dropout1': 0.3,
                'learning_rate': 0.0002
            }
        },
        'USDJPY': {
            'LSTM': {
                'units_layer1': 224,
                'recurrent_dropout': 0.15
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