
---

# Temporal Fusion Transformer (TFT) for Time Series Forecasting

## Project Overview

This project implements the **Temporal Fusion Transformer (TFT)** for time series forecasting, focusing on hourly electricity prices for France (FR) and Belgium (BE). The goal is to create a robust, accurate forecasting model by leveraging multiple data series. The project includes data preparation, feature engineering, model training, parameter tuning, and implementation for reliable forecasts that support decision-making processes.

---

## Table of Contents

1. [Project Objectives](#project-objectives)
2. [Installation](#installation)
3. [Dataset Description](#dataset-description)
4. [Methodology](#methodology)
    - [Data Preparation and Visualization](#data-preparation-and-visualization)
    - [Feature Engineering](#feature-engineering)
    - [Model Training and Cross-Validation](#model-training-and-cross-validation)
    - [Parameter Tuning and Model Optimization](#parameter-tuning-and-model-optimization)
    - [Implementation and Forecasting](#implementation-and-forecasting)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

---

## Project Objectives

- Develop a time series forecasting model using the TFT architecture.
- Forecast hourly electricity prices for France and Belgium.
- Perform data preparation, feature engineering, model training, and optimization.
- Ensure high forecast accuracy and reliability.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/tft-time-series-forecasting.git
   ```

2. Navigate to the project directory:

   ```bash
   cd tft-time-series-forecasting
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Description

The dataset includes historical electricity price data for France (FR) and Belgium (BE). Key features include:

- **Datetime**: Timestamp of each observation.
- **Electricity Price**: Target variable for prediction.
- **Weather Variables**: Exogenous variables like temperature and humidity.

---

## Methodology

### 1. Data Preparation and Visualization

- **Loading Data**: The dataset is loaded from CSV files.
- **Handling Missing Data**: Missing values are interpolated.
- **Visualization**: Price trends and seasonal patterns are visualized using line and box plots.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("electricity_prices.csv")
df.interpolate(method="linear", inplace=True)

# Visualize electricity prices
plt.plot(df['Datetime'], df['Price'])
plt.title("Electricity Price Trends")
plt.show()
```

---

### 2. Feature Engineering

- **Time Features**: Extracted date features like hour, day, and month.
- **Lag Features**: Created lag variables for past prices.
- **Scaling**: Standardization of numerical features.

```python
from sklearn.preprocessing import StandardScaler

# Feature extraction
df['hour'] = pd.to_datetime(df['Datetime']).dt.hour
df['day'] = pd.to_datetime(df['Datetime']).dt.day

# Scaling features
scaler = StandardScaler()
df['scaled_price'] = scaler.fit_transform(df[['Price']])
```

---

### 3. Model Training and Cross-Validation

- **Train-Test Split**: Data is split into training, validation, and test sets.
- **Cross-Validation**: Implemented to ensure generalization.
- **Model Initialization**: TFT model is initialized with default parameters.

```python
from sklearn.model_selection import train_test_split

# Split data
train, test = train_test_split(df, test_size=0.2, shuffle=False)
```

---

### 4. Parameter Tuning and Model Optimization

- **Grid Search**: Hyperparameter tuning using Grid Search.
- **Parameters Tuned**:
  - Learning Rate
  - Number of Layers
  - Hidden Units
  - Dropout Rate

```python
from sklearn.model_selection import GridSearchCV

# Define model and parameter grid
param_grid = {
    "learning_rate": [0.01, 0.001],
    "hidden_units": [32, 64],
    "dropout_rate": [0.1, 0.2]
}
grid_search = GridSearchCV(estimator=tft_model, param_grid=param_grid)
```

---

### 5. Implementation and Forecasting

- **Model Training**: The model is trained using the optimal parameters.
- **Forecast Generation**: Predictions for the test set are generated.
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Generate predictions
y_pred = tft_model.predict(test_features)

# Evaluate model
mae = mean_absolute_error(test_labels, y_pred)
rmse = mean_squared_error(test_labels, y_pred, squared=False)
print(f"MAE: {mae}, RMSE: {rmse}")
```

---

## Results

- **Model Performance**: MAE and RMSE scores are reported.
- **Visual Comparison**: Actual vs. predicted electricity prices are visualized.

---

## Contributing

We welcome contributions to improve the model and expand its capabilities. Follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

