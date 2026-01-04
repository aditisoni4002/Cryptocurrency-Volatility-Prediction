"""
Data Preprocessing Module for Cryptocurrency Volatility Prediction

This module handles data cleaning, normalization, and preparation for model training.
It includes missing value handling, feature scaling, and data validation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A class to handle data preprocessing for cryptocurrency volatility prediction.
    
    Methods:
    - load_data(): Load cryptocurrency data from CSV
    - handle_missing_values(): Fill missing values
    - remove_duplicates(): Remove duplicate records
    - normalize_features(): Scale numerical features
    - validate_data(): Ensure data consistency
    - preprocess(): Main preprocessing pipeline
    """
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type (str): Type of scaler - 'standard' or 'minmax'
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.df = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load cryptocurrency data from CSV file.
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            self.df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, strategy='mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy for imputation - 'mean', 'median', 'forward_fill'
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Handle missing values for numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'forward_fill':
            self.df[numerical_cols] = self.df[numerical_cols].fillna(method='ffill')
            self.df[numerical_cols] = self.df[numerical_cols].fillna(method='bfill')
        else:
            self.imputer.fit(self.df[numerical_cols])
            self.df[numerical_cols] = self.imputer.transform(self.df[numerical_cols])
        
        logger.info("Missing values handled")
        return self.df
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from dataset.
        
        Returns:
            pd.DataFrame: DataFrame without duplicates
        """
        initial_size = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_size - len(self.df)
        logger.info(f"Removed {removed} duplicate rows")
        return self.df
    
    def normalize_features(self, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Normalize numerical features using specified scaler.
        
        Args:
            columns (list): Columns to normalize. If None, normalize all numerical columns
            
        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.scaler.fit(self.df[columns])
        self.df[columns] = self.scaler.transform(self.df[columns])
        logger.info(f"Normalized {len(columns)} features")
        return self.df
    
    def validate_data(self) -> bool:
        """
        Validate data consistency and quality.
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check for remaining NaN values
        if self.df.isnull().sum().sum() > 0:
            logger.warning("Data contains NaN values")
            return False
        
        # Check for infinite values
        if np.isinf(self.df.select_dtypes(include=[np.number])).sum().sum() > 0:
            logger.warning("Data contains infinite values")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def preprocess(self, filepath: str, handle_missing=True, remove_dupes=True, 
                   normalize=True) -> pd.DataFrame:
        """
        Main preprocessing pipeline.
        
        Args:
            filepath (str): Path to data file
            handle_missing (bool): Whether to handle missing values
            remove_dupes (bool): Whether to remove duplicates
            normalize (bool): Whether to normalize features
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        self.load_data(filepath)
        
        if handle_missing:
            self.handle_missing_values()
        
        if remove_dupes:
            self.remove_duplicates()
        
        if normalize:
            self.normalize_features()
        
        self.validate_data()
        logger.info("Preprocessing complete")
        return self.df


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(scaler_type='standard')
    # df = preprocessor.preprocess('data/cryptocurrency_data.csv')
    print("Data preprocessing module loaded successfully")
