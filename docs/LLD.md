# Low-Level Design (LLD) Document
## Cryptocurrency Volatility Prediction System

### 1. Module Specifications

#### 1.1 DataPreprocessor Class (data_preprocessing.py)

**Constructor:**
```python
def __init__(self, scaler_type='standard')
```
- Initializes StandardScaler or MinMaxScaler based on scaler_type
- Initializes SimpleImputer with 'mean' strategy
- Sets df to None initially

**Key Methods:**

1. **load_data(filepath: str) -> pd.DataFrame**
   - Reads CSV file using pd.read_csv()
   - Logs file size and shape
   - Raises exception on file read error
   - Returns DataFrame

2. **handle_missing_values(strategy='mean') -> pd.DataFrame**
   - Identifies numerical columns
   - Applies mean/forward-fill imputation
   - Logs number of missing values handled
   - Returns updated DataFrame

3. **remove_duplicates() -> pd.DataFrame**
   - Counts initial rows
   - Removes duplicate rows
   - Logs number of duplicates removed
   - Returns DataFrame without duplicates

4. **normalize_features(columns=None) -> pd.DataFrame**
   - Fits scaler on specified columns
   - Transforms all data using scaler
   - Handles both StandardScaler and MinMaxScaler
   - Returns normalized DataFrame

5. **validate_data() -> bool**
   - Checks for NaN values
   - Checks for infinite values
   - Returns True if valid, False otherwise

6. **preprocess(filepath, handle_missing=True, remove_dupes=True, normalize=True) -> pd.DataFrame**
   - Orchestrates entire preprocessing pipeline
   - Executes steps in order
   - Returns final preprocessed DataFrame

#### 1.2 FeatureEngineer Class (feature_engineering.py)

**Constructor:**
```python
def __init__(self, df: pd.DataFrame)
```
- Creates copy of input DataFrame
- Initializes empty feature storage

**Key Methods:**

1. **calculate_volatility(window=20) -> pd.Series**
   - Calculates daily returns: pct_change()
   - Computes rolling standard deviation
   - Window default: 20 days
   - Returns volatility series

2. **moving_averages(windows=[20, 50, 200]) -> pd.DataFrame**
   - Creates MA for each window
   - Uses rolling().mean()
   - Returns DataFrame with MA columns

3. **bollinger_bands(window=20, num_std=2) -> pd.DataFrame**
   - Calculates moving average
   - Computes rolling std deviation
   - Upper band: MA + (num_std * std)
   - Lower band: MA - (num_std * std)
   - Returns BB DataFrame

4. **atr(window=14) -> pd.Series**
   - Calculates True Range (TR)
   - TR = max(high-low, |high-close_prev|, |low-close_prev|)
   - Returns rolling average of TR

5. **liquidity_ratio() -> pd.Series**
   - Calculates Volume / Market Cap
   - Adds small constant to avoid division by zero
   - Returns liquidity series

6. **momentum_indicators(window=14) -> pd.DataFrame**
   - Rate of Change (ROC): ((close - close_n days ago) / close_n) * 100
   - RSI: 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss
   - Returns momentum DataFrame

7. **engineer_features() -> pd.DataFrame**
   - Executes all feature engineering steps
   - Concatenates all features
   - Drops NaN values from rolling calculations
   - Returns feature-engineered DataFrame

#### 1.3 ModelTrainer Class (model_training.py)

**Constructor:**
```python
def __init__(self, test_size=0.2, random_state=42)
```
- Sets train-test split ratio
- Initializes model dictionary
- Initializes results dictionary

**Key Methods:**

1. **prepare_data(X, y) -> None**
   - Splits data using train_test_split()
   - 80% training, 20% testing
   - Logs data shape

2. **train_linear_regression() -> model**
   - Creates LinearRegression instance
   - Fits on training data
   - Stores in models dict
   - Returns trained model

3. **train_random_forest(n_estimators=100) -> model**
   - Creates RandomForestRegressor
   - Parameters: n_estimators, random_state, n_jobs=-1
   - Fits on training data
   - Returns trained model

4. **train_xgboost(n_estimators=100, learning_rate=0.1) -> model**
   - Creates XGBRegressor
   - Parameters: n_estimators, learning_rate, random_state
   - Fits on training data
   - Returns trained model

5. **train_gradient_boosting(n_estimators=100) -> model**
   - Creates GradientBoostingRegressor
   - Parameters: n_estimators, random_state
   - Fits on training data
   - Returns trained model

6. **evaluate_model(model_name) -> dict**
   - Gets model from dictionary
   - Predicts on test set
   - Calculates metrics: RMSE, MAE, R2, MAPE
   - Returns metrics dictionary

7. **train_all_models() -> dict**
   - Trains all 4 models
   - Evaluates each model
   - Returns results dictionary

8. **get_best_model() -> tuple**
   - Finds model with highest R2 score
   - Returns (model_name, model_object)

9. **save_model(model_name, filepath) -> None**
   - Serializes model using joblib.dump()
   - Saves to specified filepath

### 2. Data Flow

```
Raw CSV Data
    ↓
DataPreprocessor.load_data()
    ↓
DataPreprocessor.handle_missing_values()
    ↓
DataPreprocessor.remove_duplicates()
    ↓
DataPreprocessor.normalize_features()
    ↓
DataPreprocessor.validate_data()
    ↓
Cleaned & Normalized Data
    ↓
FeatureEngineer.calculate_volatility()    [Target Variable]
    ↓
FeatureEngineer.moving_averages()
    ↓
FeatureEngineer.bollinger_bands()
    ↓
FeatureEngineer.atr()
    ↓
FeatureEngineer.liquidity_ratio()
    ↓
FeatureEngineer.momentum_indicators()
    ↓
Featured Dataset
    ↓
ModelTrainer.prepare_data() [80-20 split]
    ↓
ModelTrainer.train_linear_regression()
ModelTrainer.train_random_forest()
ModelTrainer.train_xgboost()
ModelTrainer.train_gradient_boosting()
    ↓
ModelTrainer.evaluate_model() [for each]
    ↓
ModelTrainer.get_best_model()
    ↓
ModelTrainer.save_model()
    ↓
Trained Model (serialized)
```

### 3. Function Signatures

**Preprocessing Functions:**
- `load_data(filepath: str) -> pd.DataFrame`
- `handle_missing_values(strategy: str) -> pd.DataFrame`
- `remove_duplicates() -> pd.DataFrame`
- `normalize_features(columns: list) -> pd.DataFrame`
- `validate_data() -> bool`
- `preprocess(filepath, handle_missing, remove_dupes, normalize) -> pd.DataFrame`

**Feature Engineering Functions:**
- `calculate_volatility(window: int) -> pd.Series`
- `moving_averages(windows: list) -> pd.DataFrame`
- `bollinger_bands(window: int, num_std: float) -> pd.DataFrame`
- `atr(window: int) -> pd.Series`
- `liquidity_ratio() -> pd.Series`
- `momentum_indicators(window: int) -> pd.DataFrame`
- `engineer_features() -> pd.DataFrame`

**Model Training Functions:**
- `prepare_data(X: pd.DataFrame, y: pd.Series) -> None`
- `train_linear_regression() -> model`
- `train_random_forest(n_estimators: int) -> model`
- `train_xgboost(n_estimators: int, learning_rate: float) -> model`
- `train_gradient_boosting(n_estimators: int) -> model`
- `evaluate_model(model_name: str) -> dict`
- `train_all_models() -> dict`
- `get_best_model() -> tuple`
- `save_model(model_name: str, filepath: str) -> None`

### 4. Error Handling

- **FileNotFoundError**: Caught in load_data(), logged and raised
- **ValueError**: Raised if data not loaded before operations
- **KeyError**: Handled in model retrieval operations
- **All exceptions logged** with level INFO/ERROR

### 5. Utility Functions (utils.py)

**Planned utilities:**
- `load_model(filepath: str) -> model` - Load serialized model
- `get_metrics_summary(results: dict) -> str` - Format metrics for display
- `plot_predictions(y_true, y_pred) -> None` - Visualization helper
- `calculate_additional_metrics(y_true, y_pred) -> dict` - Extended metrics

### 6. Configuration Parameters

**Fixed Parameters:**
- Train-test split: 80-20
- CV folds: 5
- Random state: 42
- Scaler: StandardScaler (default)
- Imputation: Mean strategy
- Window sizes: 20, 50, 200 (MA); 14 (ATR, RSI)

**Tunable Parameters:**
- XGBoost: n_estimators, learning_rate, max_depth
- Random Forest: n_estimators, max_depth
- Gradient Boosting: n_estimators, learning_rate

### 7. Testing Considerations

- Unit tests for each preprocessing step
- Feature engineering output validation
- Model training convergence checks
- Metric calculation verification
- End-to-end pipeline testing

### 8. Performance Optimization

- Vectorized operations using NumPy
- Parallel processing: n_jobs=-1 for tree models
- Efficient data structures (pandas for tabular data)
- Memory-efficient imputation
- Lazy loading of models

### 9. Code Quality Standards

- PEP 8 compliance
- Type hints for all functions
- Comprehensive docstrings
- Logging at appropriate levels
- Error handling with specific exceptions
- Comments for complex logic
