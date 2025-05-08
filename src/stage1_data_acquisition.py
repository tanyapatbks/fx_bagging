"""
Stage 1: Data Acquisition
This module handles data loading, cleaning, and preprocessing for forex data.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from datetime import datetime


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
        
        # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # พยายามอ่านไฟล์ CSV โดยตรวจสอบว่าเป็น UTF-8 หรือไม่
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            # ถ้าเกิด UnicodeDecodeError ให้ลองใช้ encoding อื่น
            df = pd.read_csv(file_path, encoding='latin1')
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # แปลง Time เป็น datetime
        try:
            df['Time'] = pd.to_datetime(df['Time'])
        except ValueError as e:
            print(f"Error converting Time to datetime: {e}")
            print("Trying with different datetime formats...")
            
            # ลองใช้รูปแบบวันที่ต่างๆ
            date_formats = ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S']
            
            for date_format in date_formats:
                try:
                    df['Time'] = pd.to_datetime(df['Time'], format=date_format)
                    print(f"Successfully converted using format: {date_format}")
                    break
                except ValueError:
                    continue
            
            # ถ้ายังไม่สามารถแปลงได้ ให้แจ้งเตือน
            if not pd.api.types.is_datetime64_dtype(df['Time']):
                print("WARNING: Could not convert Time column to datetime. Using original values.")
        
        # ตรวจสอบค่า NaN และแทนที่ด้วยวิธีที่เหมาะสม
        if df.isnull().any().any():
            print(f"WARNING: Found {df.isnull().sum().sum()} NaN values in {pair} data")
            # แทนที่ค่า NaN ด้วยค่าก่อนหน้า
            df = df.fillna(method='ffill')
            # ถ้ายังมี NaN เหลืออยู่ (เช่น ในแถวแรก) ให้แทนที่ด้วยค่าถัดไป
            df = df.fillna(method='bfill')
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing datasets based on date ranges in config
        
        Args:
            df: DataFrame containing forex data
            
        Returns:
            Tuple of (training_data, testing_data)
        """
        # ตรวจสอบว่าคอลัมน์ Time เป็น datetime
        if not pd.api.types.is_datetime64_dtype(df['Time']):
            try:
                df['Time'] = pd.to_datetime(df['Time'])
            except:
                raise ValueError("Time column is not in datetime format and cannot be converted")
        
        # ตรวจสอบช่วงเวลาในข้อมูล
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
            
        print(f"Data date range: {df['Time'].min()} to {df['Time'].max()}")
        print(f"Training period: {self.config.TRAIN_START} to {self.config.TRAIN_END}")
        print(f"Testing period: {self.config.TEST_START} to {self.config.TEST_END}")
        
        # แปลงวันที่จาก string เป็น datetime object
        train_start = pd.to_datetime(self.config.TRAIN_START)
        train_end = pd.to_datetime(self.config.TRAIN_END)
        test_start = pd.to_datetime(self.config.TEST_START)
        test_end = pd.to_datetime(self.config.TEST_END)
        
        # กรองข้อมูล training
        train_mask = (df['Time'] >= train_start) & (df['Time'] <= train_end)
        train_data = df.loc[train_mask].copy().reset_index(drop=True)
        
        # กรองข้อมูล testing
        test_mask = (df['Time'] >= test_start) & (df['Time'] <= test_end)
        test_data = df.loc[test_mask].copy().reset_index(drop=True)
        
        # ตรวจสอบว่ามีข้อมูลพอหรือไม่
        if len(train_data) == 0:
            raise ValueError(f"No data found for training period: {self.config.TRAIN_START} to {self.config.TRAIN_END}")
        
        if len(test_data) == 0:
            raise ValueError(f"No data found for testing period: {self.config.TEST_START} to {self.config.TEST_END}")
        
        return train_data, test_data
    
    def check_data_quality(self, df: pd.DataFrame, name: str = "DataFrame") -> Dict[str, Union[int, str, float]]:
        """
        Check and print data quality metrics
        
        Args:
            df: DataFrame to check
            name: Name of the dataset for display purposes
            
        Returns:
            Dictionary of data quality metrics
        """
        if df is None or len(df) == 0:
            print(f"\n--- {name} is empty ---")
            return {
                "shape": (0, 0),
                "date_range": "None",
                "missing_values": 0,
                "duplicate_rows": 0,
                "quality": "Poor"
            }
        
        quality_metrics = {}
        
        print(f"\n--- {name} Quality Check ---")
        
        # Shape
        quality_metrics["shape"] = df.shape
        print(f"Shape: {df.shape}")
        
        # Date Range
        if 'Time' in df.columns:
            if pd.api.types.is_datetime64_dtype(df['Time']):
                date_range = f"{df['Time'].min()} to {df['Time'].max()}"
                quality_metrics["date_range"] = date_range
                quality_metrics["start_date"] = df['Time'].min()
                quality_metrics["end_date"] = df['Time'].max()
                print(f"Date Range: {date_range}")
            else:
                print("Warning: Time column is not in datetime format")
                quality_metrics["date_range"] = "Unknown (Time column not in datetime format)"
        else:
            print("Warning: No Time column found")
            quality_metrics["date_range"] = "Unknown (No Time column)"
        
        # Missing Values
        missing_values = df.isnull().sum().sum()
        quality_metrics["missing_values"] = missing_values
        print(f"Missing Values: {missing_values}")
        
        # Duplicate Rows
        duplicate_rows = df.duplicated().sum()
        quality_metrics["duplicate_rows"] = duplicate_rows
        print(f"Duplicate Rows: {duplicate_rows}")
        
        # Additional data quality checks
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Check for price anomalies
            price_anomalies = ~((df['High'] >= df['Open']) & 
                               (df['High'] >= df['Close']) & 
                               (df['Low'] <= df['Open']) & 
                               (df['Low'] <= df['Close']))
            anomaly_count = price_anomalies.sum()
            quality_metrics["price_anomalies"] = anomaly_count
            if anomaly_count > 0:
                print(f"WARNING: Found {anomaly_count} rows with price anomalies (High/Low vs Open/Close)")
            
            # Price ranges
            quality_metrics["price_range"] = {
                "open": (df['Open'].min(), df['Open'].max()),
                "high": (df['High'].min(), df['High'].max()),
                "low": (df['Low'].min(), df['Low'].max()),
                "close": (df['Close'].min(), df['Close'].max())
            }
            print(f"Open Price Range: {df['Open'].min()} - {df['Open'].max()}")
            print(f"Close Price Range: {df['Close'].min()} - {df['Close'].max()}")
        
        # Volume check
        if 'Volume' in df.columns:
            volume_zeros = (df['Volume'] == 0).sum()
            quality_metrics["volume_zeros"] = volume_zeros
            if volume_zeros > 0:
                print(f"WARNING: Found {volume_zeros} rows with zero volume")
            print(f"Volume Range: {df['Volume'].min()} - {df['Volume'].max()}")
        
        # Check for consistent time intervals
        if 'Time' in df.columns and pd.api.types.is_datetime64_dtype(df['Time']):
            time_diffs = df['Time'].diff().dropna()
            unique_intervals = time_diffs.unique()
            unique_intervals_str = [str(interval) for interval in unique_intervals]
            quality_metrics["time_intervals"] = unique_intervals_str
            print(f"Unique time intervals: {unique_intervals_str}")
            
            if len(unique_intervals) > 1:
                print("WARNING: Inconsistent time intervals detected!")
        
        # Add an overall quality assessment
        if missing_values == 0 and duplicate_rows == 0 and anomaly_count == 0 and volume_zeros == 0 and len(unique_intervals) == 1:
            quality_metrics["quality"] = "Excellent"
        elif missing_values < len(df) * 0.01 and duplicate_rows < len(df) * 0.01 and anomaly_count < len(df) * 0.01:
            quality_metrics["quality"] = "Good"
        elif missing_values < len(df) * 0.05 and duplicate_rows < len(df) * 0.05 and anomaly_count < len(df) * 0.05:
            quality_metrics["quality"] = "Fair"
        else:
            quality_metrics["quality"] = "Poor"
        
        print(f"Overall Data Quality: {quality_metrics['quality']}")
        
        return quality_metrics
    
    def load_all_pairs(self, pairs: Optional[List[str]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load data for all currency pairs defined in the config or the specified list
        
        Args:
            pairs: Optional list of currency pairs to load (default: use config.CURRENCY_PAIRS)
            
        Returns:
            Dictionary with structure {pair_name: {'full': full_df, 'train': train_df, 'test': test_df}}
        """
        if pairs is None:
            pairs = self.config.CURRENCY_PAIRS
        
        pair_data = {}
        quality_summary = {}
        
        for pair in pairs:
            try:
                print(f"\nProcessing {pair}...")
                
                # โหลดข้อมูล
                df = self.load_forex_data(pair)
                quality_metrics = self.check_data_quality(df, f"{pair} Full Data")
                
                # แบ่งข้อมูล
                train_data, test_data = self.split_data(df)
                train_quality = self.check_data_quality(train_data, f"{pair} Train Data")
                test_quality = self.check_data_quality(test_data, f"{pair} Test Data")
                
                # เก็บข้อมูล
                pair_data[pair] = {
                    'full': df,
                    'train': train_data,
                    'test': test_data
                }
                
                # เก็บผลการตรวจสอบคุณภาพข้อมูล
                quality_summary[pair] = {
                    'full': quality_metrics,
                    'train': train_quality,
                    'test': test_quality
                }
            except Exception as e:
                print(f"ERROR loading {pair}: {e}")
        
        # Verify all pairs have the same data dimensions
        self._verify_data_dimensions(pair_data)
        
        # Save quality summary
        self._save_quality_summary(quality_summary)
        
        return pair_data
    
    def _verify_data_dimensions(self, pair_data: Dict[str, Dict[str, pd.DataFrame]]) -> bool:
        """
        Verify that all currency pairs have consistent data dimensions
        
        Args:
            pair_data: Dictionary containing data for all currency pairs
            
        Returns:
            True if all data is consistent, False otherwise
        """
        print("\n--- Verifying Data Dimensions ---")
        
        if not pair_data:
            print("No data to verify")
            return False
        
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
            print("To align the data, you may need to:")
            print("1. Filter datasets to a common date range")
            print("2. Resample at a consistent frequency")
            print("3. Apply interpolation or padding strategies")
            return False
        
        if not test_identical:
            print("WARNING: Testing datasets have different numbers of rows!")
            return False
        
        print("All datasets have consistent dimensions")
        return True
    
    def _save_quality_summary(self, quality_summary: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Save data quality summary to a JSON file
        
        Args:
            quality_summary: Dictionary containing quality metrics for all datasets
        """
        # Convert datetime objects to strings for JSON serialization
        for pair in quality_summary:
            for dataset in quality_summary[pair]:
                if 'start_date' in quality_summary[pair][dataset]:
                    quality_summary[pair][dataset]['start_date'] = str(quality_summary[pair][dataset]['start_date'])
                if 'end_date' in quality_summary[pair][dataset]:
                    quality_summary[pair][dataset]['end_date'] = str(quality_summary[pair][dataset]['end_date'])
        
        import json
        report_path = os.path.join(self.config.LOG_PATH, "data_quality_summary.json")
        
        try:
            with open(report_path, 'w') as f:
                json.dump(quality_summary, f, indent=4)
            print(f"\nData quality summary saved to {report_path}")
        except Exception as e:
            print(f"Error saving quality summary: {e}")
    
    def resample_data(self, df: pd.DataFrame, timeframe: str = '1H') -> pd.DataFrame:
        """
        Resample data to a different timeframe
        
        Args:
            df: DataFrame containing forex data
            timeframe: Target timeframe (e.g., '1H', '4H', '1D')
            
        Returns:
            Resampled DataFrame
        """
        if not pd.api.types.is_datetime64_dtype(df['Time']):
            df['Time'] = pd.to_datetime(df['Time'])
        
        # Set time as index for resampling
        df_indexed = df.set_index('Time')
        
        # OHLC resampling
        resampled = df_indexed.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Reset index to get Time as a column again
        resampled = resampled.reset_index()
        
        print(f"Resampled data from {len(df)} rows to {len(resampled)} rows ({timeframe} timeframe)")
        
        return resampled