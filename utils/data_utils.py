"""
Data Utilities for Forex Prediction
This module contains utility functions for data handling and preprocessing.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class SequenceDataHandler:
    """
    Utility class for handling sequential data preparation for time series models
    """
    
    def __init__(self, config):
        """
        Initialize the SequenceDataHandler with configuration parameters
        
        Args:
            config: Configuration object containing sequence parameters
        """
        self.config = config
        self.scalers = {}  # Dictionary to store scalers for different datasets
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training, validation, and testing datasets based on date ranges in config
        
        Args:
            df: DataFrame containing forex data
            
        Returns:
            Tuple of (training_data, validation_data, testing_data)
        """
        # กรองข้อมูล training
        train_mask = (df['Time'] >= self.config.TRAIN_START) & (df['Time'] <= self.config.TRAIN_END)
        train_data = df.loc[train_mask].copy().reset_index(drop=True)
        
        # กรองข้อมูล validation
        val_mask = (df['Time'] >= self.config.VALIDATION_START) & (df['Time'] <= self.config.VALIDATION_END)
        val_data = df.loc[val_mask].copy().reset_index(drop=True)
        
        # กรองข้อมูล testing
        test_mask = (df['Time'] >= self.config.TEST_START) & (df['Time'] <= self.config.TEST_END)
        test_data = df.loc[test_mask].copy().reset_index(drop=True)
        
        return train_data, val_data, test_data

    def prepare_sequence_data(self, df: pd.DataFrame, target_col: str = 'Close', 
                              sequence_length: Optional[int] = None, 
                              scale_target: bool = True) -> Tuple[np.ndarray, np.ndarray, Any, int, List[str]]:
        """
        Prepare sequential data for sequence-based models (LSTM, GRU, TFT)
        
        Args:
            df: DataFrame containing forex data
            target_col: Target column name for prediction
            sequence_length: Length of sequences to create
            scale_target: Whether to scale the target values
            
        Returns:
            Tuple of (X_sequences, y_targets, scaler, target_idx, feature_columns)
        """
        if sequence_length is None:
            sequence_length = self.config.SEQUENCE_LENGTH
        
        # แยกข้อมูลที่ไม่ใช่ตัวเลข (Time)
        features = df.drop('Time', axis=1)
        feature_cols = features.columns
        
        # หาตำแหน่งของคอลัมน์เป้าหมายในชุดข้อมูล
        target_idx = list(features.columns).index(target_col)
        features = features.values
        
        # สร้าง scaler และ normalize ข้อมูล
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        
        # สร้างชุดข้อมูลในรูปแบบของ sequences
        X, y = [], []
        for i in range(len(scaled_features) - sequence_length):
            X.append(scaled_features[i:i+sequence_length])
            # ค่า y คือค่า Close ถัดไป
            if scale_target:
                y.append(scaled_features[i+sequence_length, target_idx])
            else:
                # ถ้าไม่ต้องการ scale ค่าเป้าหมาย
                y.append(features[i+sequence_length, target_idx])
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        print(f"Prepared sequence data with shape X: {X_array.shape}, y: {y_array.shape}")
        print(f"Target column '{target_col}' at index {target_idx}")
        
        return X_array, y_array, scaler, target_idx, feature_cols
    
    def prepare_tabular_data(self, df: pd.DataFrame, target_col: str = 'Close', 
                           sequence_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Any, int, List[str]]:
        """
        Prepare tabular data for tabular models (XGBoost)
        
        Args:
            df: DataFrame containing forex data
            target_col: Target column name for prediction
            sequence_length: Length of history to consider for each sample
            
        Returns:
            Tuple of (X_tabular, y_targets, scaler, target_idx, feature_columns)
        """
        if sequence_length is None:
            sequence_length = self.config.SEQUENCE_LENGTH
        
        # แยกข้อมูลที่ไม่ใช่ตัวเลข (Time)
        features = df.drop('Time', axis=1)
        feature_cols = features.columns
        
        # หาตำแหน่งของคอลัมน์เป้าหมายในชุดข้อมูล
        target_idx = list(features.columns).index(target_col)
        features = features.values
        
        # สร้าง scaler และ normalize ข้อมูล
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # สร้างชุดข้อมูลแบบ tabular โดยแต่ละแถวประกอบด้วยข้อมูลย้อนหลัง sequence_length แถว
        X, y = [], []
        for i in range(len(scaled_features) - sequence_length):
            # รวมข้อมูลจาก sequence_length แถวเป็นแถวเดียว
            row = scaled_features[i:i+sequence_length].flatten()
            X.append(row)
            # ค่า y คือค่า Close ถัดไป
            y.append(features[i+sequence_length, target_idx])
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        print(f"Prepared tabular data with shape X: {X_array.shape}, y: {y_array.shape}")
        print(f"Features flattened from {sequence_length} timesteps with {len(feature_cols)} features each")
        
        return X_array, y_array, scaler, target_idx, feature_cols
    
    def prepare_bagging_data(self, model_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare data for bagging approach by combining data from multiple currency pairs
        
        Args:
            model_data: Dictionary containing preprocessed data for different currency pairs
            
        Returns:
            Dictionary with combined data for bagging
        """
        print("\n--- Preparing Bagging Approach Data ---")
        
        # รวมข้อมูลจากทั้ง 3 คู่เงิน (ใช้ข้อมูลประเภท 'selected')
        X_seq_train_all = []
        y_seq_train_all = []
        X_seq_test_all = []
        y_seq_test_all = []
        
        X_tab_train_all = []
        y_tab_train_all = []
        X_tab_test_all = []
        y_tab_test_all = []
        
        # ข้อมูลอื่นๆ ที่ต้องเก็บไว้สำหรับแต่ละคู่เงิน
        seq_scalers = {}
        target_idxs = {}
        tab_scalers = {}
        tab_target_idxs = {}
        
        for pair in self.config.CURRENCY_PAIRS:
            pair_data = model_data[pair]['selected']
            
            print(f"Adding {pair} data to the bagging dataset")
            
            # เก็บข้อมูล sequence
            X_seq_train_all.append(pair_data['X_seq_train'])
            y_seq_train_all.append(pair_data['y_seq_train'])
            X_seq_test_all.append(pair_data['X_seq_test'])
            y_seq_test_all.append(pair_data['y_seq_test'])
            
            # เก็บข้อมูล tabular
            X_tab_train_all.append(pair_data['X_tab_train'])
            y_tab_train_all.append(pair_data['y_tab_train'])
            X_tab_test_all.append(pair_data['X_tab_test'])
            y_tab_test_all.append(pair_data['y_tab_test'])
            
            # เก็บข้อมูลอื่นๆ
            seq_scalers[pair] = pair_data['seq_scaler']
            target_idxs[pair] = pair_data['target_idx']
            tab_scalers[pair] = pair_data['tab_scaler']
            tab_target_idxs[pair] = pair_data['tab_target_idx']
        
        # รวมข้อมูล sequence
        X_seq_train_combined = np.concatenate(X_seq_train_all)
        y_seq_train_combined = np.concatenate(y_seq_train_all)
        
        # รวมข้อมูล tabular
        X_tab_train_combined = np.concatenate(X_tab_train_all)
        y_tab_train_combined = np.concatenate(y_tab_train_all)
        
        print(f"Combined sequence training data shape: X: {X_seq_train_combined.shape}, y: {y_seq_train_combined.shape}")
        print(f"Combined tabular training data shape: X: {X_tab_train_combined.shape}, y: {y_tab_train_combined.shape}")
        
        return {
            'X_seq_train': X_seq_train_combined,
            'y_seq_train': y_seq_train_combined,
            'X_seq_test_pairs': X_seq_test_all,
            'y_seq_test_pairs': y_seq_test_all,
            'X_tab_train': X_tab_train_combined,
            'y_tab_train': y_tab_train_combined,
            'X_tab_test_pairs': X_tab_test_all,
            'y_tab_test_pairs': y_tab_test_all,
            'seq_scalers': seq_scalers,
            'target_idxs': target_idxs,
            'tab_scalers': tab_scalers,
            'tab_target_idxs': tab_target_idxs
        }