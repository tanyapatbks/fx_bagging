"""
Stage 2: Feature Engineering
This module handles feature creation, transformation, and selection for forex data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


class FeatureEngineer:
    """
    Class for adding technical indicators and performing feature selection
    """
    def __init__(self, config):
        """
        Initialize the FeatureEngineer with configuration parameters
        
        Args:
            config: Configuration object containing feature engineering parameters
        """
        self.config = config
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe
        
        Args:
            df: DataFrame containing forex price data
            
        Returns:
            DataFrame with added technical indicators
        """
        # สร้าง DataFrame ใหม่เพื่อไม่ให้กระทบข้อมูลต้นฉบับ
        enhanced_df = df.copy()
        
        # ----- Trend Indicators -----
        # Simple Moving Average (SMA)
        for period in self.config.TECHNICAL_INDICATORS['sma']:
            enhanced_df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
        
        # Exponential Moving Average (EMA)
        for period in self.config.TECHNICAL_INDICATORS['ema']:
            enhanced_df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        if self.config.TECHNICAL_INDICATORS['macd']:
            enhanced_df['MACD'] = enhanced_df['EMA12'] - enhanced_df['EMA26']
            enhanced_df['MACD_Signal'] = enhanced_df['MACD'].ewm(span=9, adjust=False).mean()
            enhanced_df['MACD_Hist'] = enhanced_df['MACD'] - enhanced_df['MACD_Signal']
        
        # ----- Momentum Indicators -----
        # RSI (Relative Strength Index)
        for period in self.config.TECHNICAL_INDICATORS['rsi']:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Avoid division by zero
            avg_loss = avg_loss.replace(0, 0.001)
            
            rs = avg_gain / avg_loss
            enhanced_df[f'RSI{period}'] = 100 - (100 / (1 + rs))
        
        # Rate of Change (ROC)
        for period in self.config.TECHNICAL_INDICATORS['roc']:
            enhanced_df[f'ROC{period}'] = df['Close'].pct_change(periods=period) * 100
        
        # Momentum
        for period in self.config.TECHNICAL_INDICATORS['momentum']:
            enhanced_df[f'Momentum{period}'] = df['Close'] - df['Close'].shift(period)
        
        # ----- Volatility Indicators -----
        # Bollinger Bands
        for period in self.config.TECHNICAL_INDICATORS['bollinger']:
            enhanced_df[f'BB_Middle{period}'] = df['Close'].rolling(window=period).mean()
            enhanced_df[f'BB_Std{period}'] = df['Close'].rolling(window=period).std()
            enhanced_df[f'BB_Upper{period}'] = enhanced_df[f'BB_Middle{period}'] + (enhanced_df[f'BB_Std{period}'] * 2)
            enhanced_df[f'BB_Lower{period}'] = enhanced_df[f'BB_Middle{period}'] - (enhanced_df[f'BB_Std{period}'] * 2)
            enhanced_df[f'BB_Width{period}'] = (enhanced_df[f'BB_Upper{period}'] - enhanced_df[f'BB_Lower{period}']) / enhanced_df[f'BB_Middle{period}']
            
            # Distance of price from Bollinger Bands
            enhanced_df[f'BB_Position{period}'] = (df['Close'] - enhanced_df[f'BB_Lower{period}']) / (enhanced_df[f'BB_Upper{period}'] - enhanced_df[f'BB_Lower{period}'])
        
        # Stochastic Oscillator
        for period, d_period in zip(self.config.TECHNICAL_INDICATORS['stochastic'], [3]*len(self.config.TECHNICAL_INDICATORS['stochastic'])):
            low_period = df['Low'].rolling(window=period).min()
            high_period = df['High'].rolling(window=period).max()
            
            # ป้องกันการหารด้วย 0
            denom = high_period - low_period
            denom = denom.replace(0, 0.001)  # แทนที่ 0 ด้วยค่าเล็กๆ
            
            enhanced_df[f'Stoch_K{period}'] = 100 * ((df['Close'] - low_period) / denom)
            enhanced_df[f'Stoch_D{period}'] = enhanced_df[f'Stoch_K{period}'].rolling(window=d_period).mean()
        
        # Average True Range (ATR)
        for period in self.config.TECHNICAL_INDICATORS['atr']:
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            enhanced_df[f'ATR{period}'] = true_range.rolling(window=period).mean()
            
            # Normalized ATR (ATR divided by Close price)
            enhanced_df[f'ATR{period}_Pct'] = enhanced_df[f'ATR{period}'] / df['Close'] * 100
        
        # Average Directional Index (ADX)
        for period in self.config.TECHNICAL_INDICATORS['adx']:
            # True Range
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Directional Movement
            up_move = df['High'] - df['High'].shift()
            down_move = df['Low'].shift() - df['Low']
            
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smoothed True Range and Directional Movement
            atr = true_range.rolling(window=period).mean()
            
            # ป้องกันการหารด้วย 0
            atr_fixed = atr.replace(0, 0.001)
            
            pos_di = 100 * (pd.Series(pos_dm).rolling(window=period).mean() / atr_fixed)
            neg_di = 100 * (pd.Series(neg_dm).rolling(window=period).mean() / atr_fixed)
            
            # ป้องกันการหารด้วย 0
            denom = abs(pos_di) + abs(neg_di)
            denom = denom.replace(0, 0.001)  # แทนที่ 0 ด้วยค่าเล็กๆ
            
            # Directional Index
            dx = 100 * abs(pos_di - neg_di) / denom
            enhanced_df[f'ADX{period}'] = dx.rolling(window=period).mean()
            enhanced_df[f'DI_Positive{period}'] = pos_di
            enhanced_df[f'DI_Negative{period}'] = neg_di
        
        # ----- Price Patterns and Transformations -----
        # Candlestick Properties
        enhanced_df['Body_Size'] = abs(df['Close'] - df['Open'])
        enhanced_df['Body_Size_Pct'] = enhanced_df['Body_Size'] / df['Open'] * 100
        enhanced_df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        enhanced_df['Upper_Shadow_Pct'] = enhanced_df['Upper_Shadow'] / df['Open'] * 100
        enhanced_df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        enhanced_df['Lower_Shadow_Pct'] = enhanced_df['Lower_Shadow'] / df['Open'] * 100
        enhanced_df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
        
        # Price Momentum
        enhanced_df['Price_Change'] = df['Close'].diff()
        enhanced_df['Price_Change_Pct'] = df['Close'].pct_change() * 100
        
        # Price Acceleration (change in price change)
        enhanced_df['Price_Accel'] = enhanced_df['Price_Change'].diff()
        enhanced_df['Price_Accel_Pct'] = enhanced_df['Price_Change_Pct'].diff()
        
        # Price relative to indicators
        for period in self.config.TECHNICAL_INDICATORS['sma']:
            if f'SMA{period}' in enhanced_df.columns:
                enhanced_df[f'Price_to_SMA{period}'] = df['Close'] / enhanced_df[f'SMA{period}'] - 1
        
        # ความผันผวนของราคา
        enhanced_df['Volatility'] = df['High'] - df['Low']
        enhanced_df['Volatility_Pct'] = enhanced_df['Volatility'] / df['Close'] * 100
        
        # หาความผันผวนย้อนหลัง
        for period in [5, 10, 20]:
            enhanced_df[f'Volatility{period}'] = enhanced_df['Volatility_Pct'].rolling(window=period).std()
        
        # ปริมาณการซื้อขายสัมพัทธ์
        enhanced_df['Volume_ROC'] = df['Volume'].pct_change() * 100
        
        # Volume Moving Average
        for period in [5, 10, 20]:
            enhanced_df[f'Volume_SMA{period}'] = df['Volume'].rolling(window=period).mean()
            enhanced_df[f'Volume_Ratio{period}'] = df['Volume'] / enhanced_df[f'Volume_SMA{period}']
        
        # คุณลักษณะเกี่ยวกับเวลา
        enhanced_df['Hour'] = df['Time'].dt.hour
        enhanced_df['Day_of_Week'] = df['Time'].dt.dayofweek
        enhanced_df['Is_Weekend'] = enhanced_df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # แทนที่ค่า NaN และ Infinity ด้วย 0 (สามารถเลือกวิธีการจัดการ NaN ตามความเหมาะสม)
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.fillna(method='bfill')
        enhanced_df = enhanced_df.fillna(method='ffill')
        enhanced_df = enhanced_df.fillna(0)
        
        print(f"Added {len(enhanced_df.columns) - len(df.columns)} new features. Total features: {len(enhanced_df.columns)}")
        
        return enhanced_df
    
    # เพิ่มในโค้ด feature engineering ของคุณ
    def add_market_phase_features(df):
        # ตรวจจับการเปลี่ยนแปลงของความผันผวน
        df['Volatility20'] = df['High'].rolling(20).max() - df['Low'].rolling(20).min()
        df['Volatility_Change'] = df['Volatility20'].pct_change(5)
        
        # การเปลี่ยนเฟสของตลาด
        df['Phase_Change'] = 0
        # การเพิ่มขึ้นของความผันผวนอย่างมากมักเป็นสัญญาณของการเปลี่ยนเฟส
        df.loc[df['Volatility_Change'] > 0.5, 'Phase_Change'] = 1  
        # การลดลงอย่างมากก็อาจเป็นสัญญาณของการเปลี่ยนเฟสเช่นกัน
        df.loc[df['Volatility_Change'] < -0.3, 'Phase_Change'] = -1  
        
        return df

    def select_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                      method: Optional[str] = None, 
                      target_col: str = 'Close') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Select important features using the specified method
        
        Args:
            train_df: Training DataFrame with all features
            test_df: Testing DataFrame with all features
            method: Feature selection method ('random_forest' or 'select_from_model')
            target_col: Target column name
            
        Returns:
            Tuple of (train_df_selected, test_df_selected, importance_df)
        """
        if method is None:
            method = self.config.FEATURE_SELECTION_METHOD
            
        # แยกคุณลักษณะ (X) และค่าเป้าหมาย (y)
        X_train = train_df.drop(['Time', target_col], axis=1)
        y_train = train_df[target_col]
        
        if method == 'random_forest':
            # ใช้ Random Forest เพื่อวัดความสำคัญของคุณลักษณะ
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # ดึงค่าความสำคัญของคุณลักษณะ
            feature_importances = model.feature_importances_
            
            # สร้าง DataFrame เพื่อความสะดวกในการจัดการ
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False)
            
            # แสดงคุณลักษณะที่สำคัญที่สุด 10 อันดับแรก
            print("\nTop 10 Important Features:")
            print(importance_df.head(10))
            
            # เลือกคุณลักษณะที่สำคัญที่สุด 50%
            num_features = len(importance_df) // 2
            selected_features = importance_df.head(num_features)['Feature'].tolist()
            
            # เพิ่มคอลัมน์ Time และ target_col
            selected_features = ['Time', target_col] + selected_features
            
            # กรองข้อมูลให้เหลือเฉพาะคุณลักษณะที่เลือกไว้
            train_df_selected = train_df[selected_features]
            test_df_selected = test_df[selected_features]
            
            return train_df_selected, test_df_selected, importance_df
        
        elif method == 'select_from_model':
            # ใช้ SelectFromModel เพื่อเลือกคุณลักษณะ
            selector = SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                threshold=self.config.FEATURE_SELECTION_THRESHOLD
            )
            selector.fit(X_train, y_train)
            
            # คุณลักษณะที่ถูกเลือก
            selected_features_mask = selector.get_support()
            selected_features = X_train.columns[selected_features_mask].tolist()
            
            # แสดงคุณลักษณะที่ถูกเลือก
            print("\nSelected Features:")
            print(selected_features)
            
            # เพิ่มคอลัมน์ Time และ target_col
            selected_features = ['Time', target_col] + selected_features
            
            # กรองข้อมูลให้เหลือเฉพาะคุณลักษณะที่เลือกไว้
            train_df_selected = train_df[selected_features]
            test_df_selected = test_df[selected_features]
            
            # สร้าง DataFrame ของความสำคัญ
            # ในกรณีนี้เราไม่มีค่าความสำคัญโดยตรง จึงให้ค่าเท่ากันหมด
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': [1.0 if feature in selected_features else 0.0 
                              for feature in X_train.columns]
            }).sort_values('Importance', ascending=False)
            
            return train_df_selected, test_df_selected, importance_df
        
        else:
            print(f"Method {method} not implemented. Returning original DataFrames.")
            # Create a dummy importance_df
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': [1.0] * len(X_train.columns)
            }).sort_values('Importance', ascending=False)
            return train_df, test_df, importance_df