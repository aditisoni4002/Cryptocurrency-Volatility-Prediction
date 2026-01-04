"""
Utility Functions for Cryptocurrency Volatility Prediction

This module provides helper functions for model loading, metric calculations,
visualization, and data handling throughout the project.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

def load_model(filepath: str) -> Any:
    """
    Load a serialized model from file.
    
    Args:
        filepath (str): Path to saved model file
        
    Returns:
        Trained model object
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded successfully from {filepath}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {filepath}")
        raise

def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to file.
    
    Args:
        model: Trained model object
        filepath (str): Path to save model
    """
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")

def get_metrics_summary(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format model evaluation results for display.
    
    Args:
        results (dict): Dictionary of model results
        
    Returns:
        str: Formatted summary string
    """
    summary = "\n=== Model Performance Summary ===\n"
    for model_name, metrics in results.items():
        summary += f"\n{model_name}:\n"
        for metric, value in metrics.items():
            summary += f"  {metric}: {value:.6f}\n"
    return summary

def calculate_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate additional evaluation metrics.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae
    }

def normalize_data(data: pd.DataFrame, columns: list, scaler: Any = None) -> Tuple[pd.DataFrame, Any]:
    """
    Normalize specified columns in dataframe.
    
    Args:
        data (pd.DataFrame): Input dataframe
        columns (list): Columns to normalize
        scaler: Sklearn scaler object (optional)
        
    Returns:
        tuple: (normalized dataframe, scaler object)
    """
    from sklearn.preprocessing import StandardScaler
    
    if scaler is None:
        scaler = StandardScaler()
        data[columns] = scaler.fit_transform(data[columns])
    else:
        data[columns] = scaler.transform(data[columns])
    
    return data, scaler

def split_time_series(data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data maintaining temporal order.
    
    Args:
        data (pd.DataFrame): Time series data
        train_ratio (float): Training set ratio
        
    Returns:
        tuple: (train_data, test_data)
    """
    split_point = int(len(data) * train_ratio)
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    logger.info(f"Data split: Train {len(train_data)}, Test {len(test_data)}")
    return train_data, test_data

def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """
    Validate prediction results.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        bool: True if valid
    """
    if len(y_true) != len(y_pred):
        logger.warning("Length mismatch between y_true and y_pred")
        return False
    
    if np.any(np.isnan(y_pred)):
        logger.warning("NaN values found in predictions")
        return False
    
    if np.any(np.isinf(y_pred)):
        logger.warning("Infinite values found in predictions")
        return False
    
    return True

def get_feature_importance(model: Any, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Trained tree-based model
        feature_names (list): Feature names
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return importance_df
    
    logger.warning("Model does not support feature importance extraction")
    return None
