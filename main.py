"""
Main Pipeline for Cryptocurrency Volatility Prediction

This script orchestrates the entire ML pipeline from data preprocessing
through model training and evaluation.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from utils import get_metrics_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main execution function for the complete pipeline.
    """
    logger.info("="*60)
    logger.info("Starting Cryptocurrency Volatility Prediction Pipeline")
    logger.info("="*60)
    
    # ============ STEP 1: DATA PREPROCESSING ============
    logger.info("\nStep 1: Data Preprocessing")
    logger.info("-" * 40)
    
    try:
        preprocessor = DataPreprocessor(scaler_type='standard')
        # Load sample data path (update with actual data path)
        data_path = 'data/cryptocurrency_data.csv'
        
        if os.path.exists(data_path):
            df = preprocessor.preprocess(
                data_path,
                handle_missing=True,
                remove_dupes=True,
                normalize=True
            )
            logger.info(f"Preprocessed data shape: {df.shape}")
        else:
            logger.warning(f"Data file not found: {data_path}")
            logger.info("Please provide cryptocurrency data in data/cryptocurrency_data.csv")
            return
            
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        return
    
    # ============ STEP 2: FEATURE ENGINEERING ============
    logger.info("\nStep 2: Feature Engineering")
    logger.info("-" * 40)
    
    try:
        engineer = FeatureEngineer(df)
        df_features = engineer.engineer_features()
        logger.info(f"Engineered features shape: {df_features.shape}")
        logger.info(f"Features created: {len(df_features.columns)} columns")
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        return
    
    # ============ STEP 3: PREPARE DATA FOR MODELING ============
    logger.info("\nStep 3: Data Preparation for Modeling")
    logger.info("-" * 40)
    
    try:
        # Extract features and target
        feature_cols = [col for col in df_features.columns if col != 'volatility']
        X = df_features[feature_cols]
        y = df_features['volatility']
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target variable shape: {y.shape}")
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        return
    
    # ============ STEP 4: MODEL TRAINING ============
    logger.info("\nStep 4: Model Training and Evaluation")
    logger.info("-" * 40)
    
    try:
        trainer = ModelTrainer(test_size=0.2, random_state=42)
        trainer.prepare_data(X, y)
        
        results = trainer.train_all_models()
        
        # Display results
        logger.info("\n" + get_metrics_summary(results))
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return
    
    # ============ STEP 5: MODEL SELECTION AND SAVING ============
    logger.info("\nStep 5: Model Selection and Saving")
    logger.info("-" * 40)
    
    try:
        best_model_name, best_model = trainer.get_best_model()
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best R2 Score: {results[best_model_name]['R2_Score']:.6f}")
        
        # Save best model
        model_path = os.path.join('models', f'{best_model_name.lower()}_model.pkl')
        os.makedirs('models', exist_ok=True)
        trainer.save_model(best_model_name, model_path)
        logger.info(f"Model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"Error in model selection: {str(e)}")
        return
    
    # ============ COMPLETION ============
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)
    logger.info(f"\nNext Steps:")
    logger.info(f"1. Review results in the logs above")
    logger.info(f"2. Deploy model using: streamlit run app/app.py")
    logger.info(f"3. Check documentation in docs/ folder")
    logger.info("="*60)

if __name__ == "__main__":
    main()
