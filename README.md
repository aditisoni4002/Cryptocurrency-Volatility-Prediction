# Cryptocurrency Volatility Prediction

## Overview
This project builds a machine learning model to predict cryptocurrency volatility levels based on historical market data. The model utilizes OHLC (Open, High, Low, Close) prices, trading volume, and market capitalization to forecast periods of heightened volatility, enabling traders and financial institutions to manage risks effectively.

## Problem Statement
Cryptocurrency markets are highly volatile, making accurate volatility prediction crucial for:
- Risk management
- Portfolio allocation
- Developing trading strategies
- Informed decision-making

## Project Structure
```
Cryptocurrency-Volatility-Prediction/
│
├── data/
│   └── cryptocurrency_data.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
│
├── models/
│   └── trained_model.pkl
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── docs/
│   ├── HLD.md
│   ├── LLD.md
│   ├── Pipeline_Architecture.md
│   └── Final_Report.md
│
├── README.md
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository:
```bash
git clone https://github.com/aditisoni4002/Cryptocurrency-Volatility-Prediction.git
cd Cryptocurrency-Volatility-Prediction
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
- **Source**: Cryptocurrency historical price data
- **Coverage**: 50+ cryptocurrencies
- **Features**: Date, Symbol, Open, High, Low, Close, Volume, Market Cap
- **Frequency**: Daily records

## Key Features Engineered
- Moving averages (20, 50, 200 days)
- Rolling volatility
- Liquidity ratios (Volume/Market Cap)
- Bollinger Bands
- ATR (Average True Range)
- Price momentum indicators

## Models Implemented
1. Linear Regression
2. Random Forest
3. Gradient Boosting (XGBoost)
4. LSTM (Long Short-Term Memory) for time-series forecasting

## Model Performance
Evaluation metrics used:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

## Deployment
The trained model is deployed using **Streamlit** for interactive predictions:

```bash
cd app
streamlit run app.py
```

Access the application at `http://localhost:8501`

## Usage
1. Prepare your data in CSV format with required columns
2. Run data preprocessing script
3. Execute model training
4. Use the Streamlit app for predictions
5. Review visualizations and metrics

## Results
- Achieved **R² Score**: [To be updated after training]
- **Best Model**: [To be updated]
- **Key Insights**: [To be updated in final report]

## Documentation
- **HLD Document**: System architecture overview
- **LLD Document**: Detailed component implementation
- **Pipeline Architecture**: Data flow and processing pipeline
- **Final Report**: Comprehensive findings and insights

## Contributors
- Aditisoni4002

## License
MIT License

## Contact
For questions or collaboration, please open an issue on GitHub.
