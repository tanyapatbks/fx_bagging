"""
Stage 1: Data Acquisition
This module handles data loading, cleaning, and preprocessing for forex data.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class DataLoader:
    """
    Class for loading, preprocessing, and splitting forex data
    """
    def __init__(self, config):
        """
        Initialize the DataLoader with configuration parameters
        
        Args:
            config: Configuration object containing data paths and parameters
        """
        self.config = config
        config.create_directories()
    
    def load_forex_data(self, pair: str) -> pd.DataFrame:
        """
        Load forex data from CSV file for a specific currency pair
        
        Args:
            pair: Currency pair code (e.g., "EURUSD")
            
        Returns:
            DataFrame containing the forex data
        """
        file_path = os.path.join(self.config.DATA_PATH, f"{pair}_1H.csv")
        df = pd.read_csv(file_path)
        
        # แปลง Time เป็น datetime
        df['Time'] = pd.to_datetime(df['Time'])
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing datasets based on date ranges in config
        
        Args:
            df: DataFrame containing forex data
            
        Returns:
            Tuple of (training_data, testing_data)
        """
        # กรองข้อมูล training
        train_mask = (df['Time'] >= self.config.TRAIN_START) & (df['Time'] <= self.config.TRAIN_END)
        train_data = df.loc[train_mask].copy().reset_index(drop=True)
        
        # กรองข้อมูล testing
        test_mask = (df['Time'] >= self.config.TEST_START) & (df['Time'] <= self.config.TEST_END)
        test_data = df.loc[test_mask].copy().reset_index(drop=True)
        
        return train_data, test_data
    
    def check_data_quality(self, df: pd.DataFrame, name: str = "DataFrame") -> None:
        """
        Check and print data quality metrics
        
        Args:
            df: DataFrame to check
            name: Name of the dataset for display purposes
        """
        print(f"\n--- {name} Quality Check ---")
        print(f"Shape: {df.shape}")
        print(f"Date Range: {df['Time'].min()} to {df['Time'].max()}")
        print(f"Missing Values: {df.isnull().sum().sum()}")
        print(f"Duplicate Rows: {df.duplicated().sum()}")
        
        # Additional data quality checks
        print(f"Open Price Range: {df['Open'].min()} - {df['Open'].max()}")
        print(f"Close Price Range: {df['Close'].min()} - {df['Close'].max()}")
        print(f"Volume Range: {df['Volume'].min()} - {df['Volume'].max()}")
        
        # Check for consistent time intervals
        time_diffs = df['Time'].diff().dropna()
        unique_intervals = time_diffs.unique()
        print(f"Unique time intervals: {[str(interval) for interval in unique_intervals]}")
        
        if len(unique_intervals) > 1:
            print("WARNING: Inconsistent time intervals detected!")
    
    def load_all_pairs(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load data for all currency pairs defined in the config
        
        Returns:
            Dictionary with structure {pair_name: {'full': full_df, 'train': train_df, 'test': test_df}}
        """
        pair_data = {}
        
        for pair in self.config.CURRENCY_PAIRS:
            print(f"\nProcessing {pair}...")
            
            # โหลดข้อมูล
            df = self.load_forex_data(pair)
            self.check_data_quality(df, f"{pair} Full Data")
            
            # แบ่งข้อมูล
            train_data, test_data = self.split_data(df)
            self.check_data_quality(train_data, f"{pair} Train Data")
            self.check_data_quality(test_data, f"{pair} Test Data")
            
            # เก็บข้อมูล
            pair_data[pair] = {
                'full': df,
                'train': train_data,
                'test': test_data
            }
        
        # Verify all pairs have the same data dimensions
        self._verify_data_dimensions(pair_data)
        
        return pair_data
    
    def _verify_data_dimensions(self, pair_data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        """
        Verify that all currency pairs have consistent data dimensions
        
        Args:
            pair_data: Dictionary containing data for all currency pairs
        """
        print("\n--- Verifying Data Dimensions ---")
        
        # Check training data
        train_shapes = {pair: data['train'].shape for pair, data in pair_data.items()}
        print("Training data shapes:")
        for pair, shape in train_shapes.items():
            print(f"  {pair}: {shape}")
        
        # Check testing data
        test_shapes = {pair: data['test'].shape for pair, data in pair_data.items()}
        print("Testing data shapes:")
        for pair, shape in test_shapes.items():
            print(f"  {pair}: {shape}")
        
        # Check if all pairs have the same shapes
        train_identical = len(set(shape[0] for shape in train_shapes.values())) == 1
        test_identical = len(set(shape[0] for shape in test_shapes.values())) == 1
        
        if not train_identical:
            print("WARNING: Training datasets have different numbers of rows!")
            # Here you might want to implement a strategy to align the data
        
        if not test_identical:
            print("WARNING: Testing datasets have different numbers of rows!")
            # Here you might want to implement a strategy to align the data