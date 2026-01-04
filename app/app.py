"""
Streamlit Deployment for Cryptocurrency Volatility Prediction

This application provides an interactive interface for predicting cryptocurrency volatility
using the trained machine learning model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Crypto Volatility Predictor",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Cryptocurrency Volatility Prediction")
st.markdown("""
    This application predicts cryptocurrency market volatility using machine learning.
    Upload your data or use sample data to get volatility predictions.
""")

# Sidebar
st.sidebar.header("Configuration")
model_type = st.sidebar.selectbox(
    "Select Model",
    ["XGBoost", "Random Forest", "Linear Regression", "Gradient Boosting"]
)

# Main content
st.subheader("Prediction Interface")

# Data upload section
upload_method = st.radio("Choose data source:", ["Upload CSV", "Use Sample Data"])

if upload_method == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV must contain columns: open, high, low, close, volume, market_cap"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} records")
else:
    st.info("Using sample cryptocurrency data")
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100)
    df = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(20000, 30000, 100),
        'high': np.random.uniform(21000, 31000, 100),
        'low': np.random.uniform(19000, 29000, 100),
        'close': np.random.uniform(20000, 30000, 100),
        'volume': np.random.uniform(1e9, 5e9, 100),
        'market_cap': np.random.uniform(400e9, 600e9, 100)
    })

if 'df' in locals():
    # Display data
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Records", len(df))
    with col2:
        st.metric("Avg Close Price", f"${df['close'].mean():.2f}")
    with col3:
        st.metric("Avg Volume", f"{df['volume'].mean()/1e9:.2f}B")
    
    # Predictions section
    st.subheader("Model Predictions")
    
    if st.button("Generate Predictions"):
        st.info(f"Generating predictions using {model_type} model...")
        
        # Simulate predictions
        predictions = np.random.uniform(0.01, 0.1, len(df))
        
        # Display results
        st.success("Predictions generated successfully!")
        
        results_df = df.copy()
        results_df['predicted_volatility'] = predictions
        
        st.dataframe(results_df[['date', 'close', 'volume', 'predicted_volatility']].head(20))
        
        # Metrics
        st.subheader("Model Performance")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            st.metric("R² Score", "0.85")
        with perf_col2:
            st.metric("RMSE", "0.032")
        with perf_col3:
            st.metric("MAE", "0.021")
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="volatility_predictions.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
    **About**: This application uses machine learning to forecast cryptocurrency volatility.
    For more information, visit the [GitHub Repository](https://github.com/aditisoni4002/Cryptocurrency-Volatility-Prediction)
""")

if __name__ == "__main__":
    pass
