"""
Feature Engineering Module for Cryptocurrency Volatility Prediction

This module creates new features from raw OHLC data to improve model performance.
Includes technical indicators, volatility measures, and liquidity ratios.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering for cryptocurrency volatility prediction.
    
    Methods:
    - calculate_volatility(): Rolling volatility
    - moving_averages(): Calculate MA
    - bollinger_bands(): Calculate BB
    - atr(): Average True Range
    - liquidity_ratio(): Volume/Market Cap
    - momentum_indicators(): Price momentum
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def calculate_volatility(self, window: int = 20) -> pd.Series:
        """
        Calculate rolling volatility (standard deviation of returns).
        
        Args:
            window (int): Window size for rolling calculation
            
        Returns:
            pd.Series: Volatility values
        """
        returns = self.df['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        return volatility
    
    def moving_averages(self, windows: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """
        Calculate moving averages.
        
        Args:
            windows (List[int]): Window sizes
            
        Returns:
            pd.DataFrame: MA values
        """
        ma_df = pd.DataFrame()
        for window in windows:
            ma_df[f'MA_{window}'] = self.df['close'].rolling(window=window).mean()
        return ma_df
    
    def bollinger_bands(self, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            window (int): Window size
            num_std (float): Number of standard deviations
            
        Returns:
            pd.DataFrame: BB values
        """
        ma = self.df['close'].rolling(window=window).mean()
        std = self.df['close'].rolling(window=window).std()
        
        bb_df = pd.DataFrame()
        bb_df['BB_upper'] = ma + (std * num_std)
        bb_df['BB_lower'] = ma - (std * num_std)
        bb_df['BB_middle'] = ma
        return bb_df
    
    def atr(self, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            window (int): Window size
            
        Returns:
            pd.Series: ATR values
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    def liquidity_ratio(self) -> pd.Series:
        """
        Calculate liquidity ratio (Volume/Market Cap).
        
        Returns:
            pd.Series: Liquidity ratio
        """
        if 'volume' in self.df.columns and 'market_cap' in self.df.columns:
            liquidity = self.df['volume'] / (self.df['market_cap'] + 1)
            return liquidity
        return pd.Series([0] * len(self.df))
    
    def momentum_indicators(self, window: int = 14) -> pd.DataFrame:
        """
        Calculate momentum indicators (ROC, RSI proxy).
        
        Args:
            window (int): Window size
            
        Returns:
            pd.DataFrame: Momentum indicators
        """
        close = self.df['close']
        
        # Rate of Change
        roc = ((close - close.shift(window)) / close.shift(window)) * 100
        
        # RSI-like calculation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        mom_df = pd.DataFrame()
        mom_df['ROC'] = roc
        mom_df['RSI'] = rsi
        return mom_df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Execute all feature engineering.
        
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        # Add volatility
        self.df['volatility'] = self.calculate_volatility()
        
        # Add moving averages
        ma = self.moving_averages()
        self.df = pd.concat([self.df, ma], axis=1)
        
        # Add Bollinger Bands
        bb = self.bollinger_bands()
        self.df = pd.concat([self.df, bb], axis=1)
        
        # Add ATR
        self.df['ATR'] = self.atr()
        
        # Add liquidity
        self.df['liquidity_ratio'] = self.liquidity_ratio()
        
        # Add momentum
        mom = self.momentum_indicators()
        self.df = pd.concat([self.df, mom], axis=1)
        
        # Drop NaN values created by rolling calculations
        self.df = self.df.dropna()
        
        logger.info(f"Features engineered. Shape: {self.df.shape}")
        return self.df


if __name__ == "__main__":
    print("Feature engineering module loaded successfully")
