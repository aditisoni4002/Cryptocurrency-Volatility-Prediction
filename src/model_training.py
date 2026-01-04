"""
Model Training Module for Cryptocurrency Volatility Prediction

This module trains multiple machine learning models including Linear Regression,
Random Forest, XGBoost, and LSTM for volatility prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import logging
from typing import Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Train multiple models for volatility prediction.
    
    Models:
    - Linear Regression
    - Random Forest
    - Gradient Boosting (XGBoost)
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Split data into train and test sets.
        
        Args:
            X: Features dataframe
            y: Target series
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        logger.info(f"Data split: Train {self.X_train.shape}, Test {self.X_test.shape}")
    
    def train_linear_regression(self) -> Any:
        """
        Train Linear Regression model.
        
        Returns:
            Trained model
        """
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = model
        logger.info("Linear Regression trained")
        return model
    
    def train_random_forest(self, n_estimators: int = 100) -> Any:
        """
        Train Random Forest model.
        
        Args:
            n_estimators: Number of trees
            
        Returns:
            Trained model
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        logger.info("Random Forest trained")
        return model
    
    def train_xgboost(self, n_estimators: int = 100, learning_rate: float = 0.1) -> Any:
        """
        Train XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            
        Returns:
            Trained model
        """
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        logger.info("XGBoost trained")
        return model
    
    def train_gradient_boosting(self, n_estimators: int = 100) -> Any:
        """
        Train Gradient Boosting model.
        
        Args:
            n_estimators: Number of boosting stages
            
        Returns:
            Trained model
        """
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = model
        logger.info("Gradient Boosting trained")
        return model
    
    def evaluate_model(self, model_name: str) -> Dict[str, float]:
        """
        Evaluate trained model on test set.
        
        Args:
            model_name: Name of model to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'MAPE': mape
        }
        
        self.results[model_name] = metrics
        logger.info(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        return metrics
    
    def train_all_models(self) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate all models.
        
        Returns:
            Dictionary with all results
        """
        self.train_linear_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.train_gradient_boosting()
        
        for model_name in self.models.keys():
            self.evaluate_model(model_name)
        
        return self.results
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get model with best R2 score.
        
        Returns:
            Tuple of (model_name, model_object)
        """
        best_model = max(self.results, key=lambda x: self.results[x]['R2_Score'])
        return best_model, self.models[best_model]
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            model_name: Name of model to save
            filepath: Path to save model
        """
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model saved: {filepath}")


if __name__ == "__main__":
    print("Model training module loaded successfully")
