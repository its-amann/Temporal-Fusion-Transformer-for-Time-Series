<p align="center">
  <img src="time_series_forecasting.png" alt="Time Series Forecasting with TFT" width="400"/>
</p>

<h1 align="center"> 📈 Time Series Forecasting with Temporal Fusion Transformer (TFT) 📈</h1>

<p align="center">
  <strong>A Deep Learning approach to multi-series time series forecasting using the Temporal Fusion Transformer model in Python with the Darts library.</strong>
</p>

<p align="center">
  <a href="https://github.com/your-username/your-repo">
    <img src="https://img.shields.io/github/license/your-username/your-repo.svg" alt="License">
  </a>
  <a href="https://github.com/your-username/your-repo/issues">
    <img src="https://img.shields.io/github/issues/your-username/your-repo.svg" alt="Issues">
  </a>
  <a href="https://github.com/your-username/your-repo/stargazers">
    <img src="https://img.shields.io/github/stars/your-username/your-repo.svg" alt="Stars">
  </a>
</p>

---

## 🚀 Overview

Welcome to the **Time Series Forecasting with TFT**, a powerful project designed to perform multi-series time series forecasting using the cutting-edge Temporal Fusion Transformer (TFT) model. This project utilizes the `darts` library in Python, a user-friendly package built for time series analysis and forecasting, and runs on Google Colab, taking full advantage of its GPU capabilities. 

The core of this project is the implementation of the Temporal Fusion Transformer, a deep learning model known for its superior performance in time series tasks, particularly when dealing with multiple series and complex temporal dynamics. This project offers a complete end-to-end example, handling data loading, preprocessing, model training, cross-validation, parameter tuning, and future forecasting.

✨ **Wow Factors:**

- **Temporal Fusion Transformer Model:** Leverages the state-of-the-art TFT model known for robust and accurate forecasting.
- **Multi-Series Forecasting:** Capable of handling and predicting multiple time series simultaneously, ideal for complex datasets.
- **Automated Parameter Tuning:** Implements parameter optimization using a randomized search for peak model performance.
- **GPU Acceleration on Google Colab:** Utilizes Google Colab's GPU to significantly reduce training time, enhancing efficiency.
- **End-to-End Workflow:** Includes complete pipeline from data loading, preprocessing, model training, validation and forecasting.
- **Darts Library:** Uses `darts`, a user-friendly Python package, which streamlines time series tasks.
- **Detailed Data Preprocessing:** Implements scaling of the target variable and past covariates, along with time-based encoders.

<p align="center">
  <img src="tft_architecture.png" alt="TFT Architecture" width="400"/>
</p>

---

## 🧰 Table of Contents
- [🚀 Overview](#-overview)
- [🛠 Features](#-features)
- [📸 Screenshots](#-screenshots)
- [🔧 Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
- [💻 Usage](#-usage)
- [🗂️ Codebase Overview](#️-codebase-overview)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
---

## 🛠 Features

- **Data Loading and Visualization:** Load time series data from CSV files and visualize the time series to understand data patterns.
- **Multi-Series Time Series Creation:** Convert DataFrame into time series objects using the `darts` library, for both target variable and covariates.
- **Static Covariate Handling:** Utilize a unique column as the static covariate to differentiate the multiple series
- **Custom Time Encoders:** Apply cyclic, datetime, position and year encoders to capture temporal characteristics effectively.
- **Data Scaling:** Scale the target variable and past/future covariates for effective neural network training.
- **TFT Model Training:** Set up and train the Temporal Fusion Transformer model with specified parameters, such as layer size, attention heads, dropout and batch size.
- **Cross-Validation:** Implement time series cross-validation with a rolling forecasting window.
- **Automated Parameter Tuning:** Perform parameter tuning through a randomized search, including multiple combinations of learning rate, input length, output length, dropout rates, number of attention heads, etc.
- **RMSE Evaluation:** Evaluate the model's performance by calculating the Root Mean Squared Error (RMSE) on the validation set.
- **Future Forecasting:** Predict future values for multiple time series using the optimized TFT model.

---

## 📸 Screenshots

<p align="center">
<img src="time_series.png" alt="Time series plot" width="600"/>
</p>
<p align="center">
   <font size="5"><b>
    Multi Time Series plot of the target variable for all the series
    </b></font>
</p>

<p align="center">
<img src="parameter_tuning.png" alt="parameter tunning" width="600"/>
</p>
<p align="center">
   <font size="5"><b>
        RMSE results of parameter tunning
    </b></font>
</p>

---

## 🔧 Installation

### Prerequisites

-   **Google Colab Account:** A Google Colab account is required to run the notebook environment with GPU acceleration.
-   **Python 3.x:**  Python 3.x is needed to run the `darts` library and other python scripts.
-   **Darts Library:** This is the core library used for this project. 


To install the darts library run:

`pip install -q darts`


- **pandas, matplotlib, statsmodels and scikit-learn:** Install with pip:

`
pip install pandas matplotlib statsmodels scikit-learn
`

### Steps

1.  **Clone the Repository:**

    `
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    `
2.  **Open the notebook:** Open the `time_series_forecasting_tft.ipynb` file in google colab.
3.  **Mount Google Drive:** Mount your Google Drive to load the datasets and save files by running the following code in google colab:
    `
    from google.colab import drive
    drive.mount('/content/drive')
    `
4.  **Change directory:** Set the working directory to your project folder using the following code:
    `
    %cd /content/drive/MyDrive/your-project-folder
    `
    > Replace `/content/drive/MyDrive/your-project-folder` with your project folder.
5.  **Run the notebook:** Run all cells of the notebook to execute all steps of the code, which will load the data, preprocess, train the model and predict the future.

---

## 💻 Usage

This project provides a complete end-to-end workflow that can be followed sequentially. The Google Colab notebook is organized into several sections, each performing a specific task, explained below:

1.  **Libraries and Data Loading:**
    -   Import all necessary libraries.
    -   Mount Google Drive to access your data files.
    -   Change the working directory to your project folder.
    -   Install the `darts` library.
    -   Load the hourly electricity consumption data from the `electricity.csv` file using `pandas`.
    -   Visualize the target variable for each unique ID (each series).
    -   Filter out the specific time series we want to perform the modelling on, which are "BE" and "FR".
2.  **Time Series, Time and Static Covariates:**
    - Create custom time-related encoders using the `encode_year` function and define all the encoders to add through a dictionary called `add_encoders`.
    -  Convert the "unique_id" column into numerical codes for the time series objects.
    - Reset the index of your dataframes so that the time indices become a column named `ds`.
    - Define the column names of the time and target variables, and the columns names of the series identifiers in the variable `GROUP_COL`.
    - Create the time series for the target variable, and past and future covariates.
3. **Scaling**
  - Scale the target variable, past and future covariates to prepare the data for the TFT model.
4.  **TFT Model Configuration and Training:**
    - Define the forecasting horizon (number of time points to predict).
    - Configure the TFT model with specified parameters.
    - Fit the model with the transformed target and covariates.
5.  **Cross-Validation:**
    - Perform cross-validation to assess model performance.
    - Calculate the RMSE during cross validation and prints the mean of all the backtesting periods.
6.  **Parameter Tuning:**
    - Define the ranges for the model's hyperparameters.
    - Define the fixed parameters to be used across all different configurations.
    - Tune model parameters using a random search.
    - Save the best model parameters to a CSV file for future use.
7.  **Future Predictions:**
    - Load the best parameters from the CSV file.
    - Train the final model with the best parameters on the entire dataset.
    - Generate forecasts for the future with the trained model.
    - Unscale the predicted values using the inverse transform.
    - Export the predictions to a CSV file.

---

## 🗂️ Codebase Overview

This project's codebase is structured around a single Jupyter Notebook file, `time_series_forecasting_tft.ipynb`, which encompasses all steps of the time series forecasting workflow. Here's a breakdown of the key sections and their purpose:

1.  **Libraries and Data Loading:**
    -   **Purpose:** This section initializes the project by importing necessary libraries, mounting Google Drive, changing the working directory and loading the input data, including time series and covariate data.
    -   **Libraries:**
         -  `pandas`: for data loading and manipulation.
        -   `matplotlib.pyplot`: for data visualization.
        -   `statsmodels.graphics.tsaplots`: for time series analysis plots.
        -   `statsmodels.tsa.seasonal`: for time series decomposition.
        -  `sklearn.metrics`: for model evaluation metrics.
        -   `numpy`: for numerical operations.
        -   `sklearn.model_selection`: for parameter tuning.
        -  `darts.timeseries`: for creating and handling time series objects.
        -  `darts.utils.timeseries_generation`: for creating time-related series.
        - `darts.dataprocessing.transformers`: for scaling and transforming data.
        - `darts.models`: for the TFT model.
     -   **Design Choices:** Libraries are loaded at the beginning for ease of access. Using `pandas` for data manipulation ensures that both training and forecasting data are ready for consumption.

2.  **Time Series, Time and Static Covariates:**
    -   **Purpose:** Transform and prepare time-related and static data for the TFT model. This include creating multiple time series, encoding the time information, scaling the data, etc.
    -  **Key Components:**
        - `encode_year`: A custom function that encodes the year as a normalized value relative to 2000.
        - `add_encoders`: a dictionary that stores parameters for the time encoders.
        - `TimeSeries.from_group_dataframe`: for transforming a pandas data frame into a time series object.
    -   **Design Choices:** Categorical encoding is applied to the 'unique_id' for the static covariates, allowing the TFT model to differentiate between the various series. Data is converted into darts time series object, since this is how the model receives the inputs.

3.  **Scaling:**
    -  **Purpose:** Prepare the time series variables to be consumed by the TFT model, by scaling the target, past and future covariates.
    -  **Key Components:**
          - `Scaler`: A class from `darts` to apply the scaling.
    -  **Design Choices:** Scaling data helps ensure that the deep learning algorithm can converge properly.

4.  **TFT Model Configuration and Training:**
    -  **Purpose:** Set up and train the TFT model. Includes setting the training parameters, using GPU, configuring the data, and the model's architecture, optimizing the performance of the model.
    -   **Key Components:**
        -    `TFTModel`: Model from `darts` to use the Temporal Fusion Transformer model.
       -   Model parameters: `input_chunk_length`, `output_chunk_length`, `hidden_size`, `lstm_layers`, `num_attention_heads`, `dropout`, `batch_size`, `n_epochs`, and `use_static_covariates` among others.
       - `pl_trainer_kwargs`: A dictionary used to pass parameters to the PyTorch Lightning trainer, such as setting the accelerator to GPU and the devices to use.
    -   **Design Choices:** Key parameters are set for the model, such as the input and output window length, number of LSTM layers, and dropout, balancing the model's complexity and ability to generalize.

5.  **Cross-Validation:**
    -   **Purpose:** Evaluate the model performance using time series cross-validation, simulating real world forecasting scenarios by using a rolling window.
    -   **Key Components:**
        -   `historical_forecasts`: Function to perform the cross-validation using a rolling window.
        -  RMSE Calculation: Calculate RMSE for each series and for all the backtesting periods.
    -   **Design Choices:** Cross-validation was implemented for the sake of robust model evaluation, especially with the nature of time series data, where time order must be respected.

6.  **Parameter Tuning:**
    -   **Purpose:** Optimize the TFT model parameters for optimal model performance and generalization. This is done using a randomized search and a parameter grid defined by the user.
    -   **Key Components:**
          -  `ParameterSampler`: generates the random parameters for each run from the provided ranges.
          -    `TFTModel`: Trains the model with different parameters.
          -    RMSE Calculation: evaluate the model in each different configuration.
    -   **Design Choices:** Randomized search was used because it is a good way to quickly explore a wide range of possible parameters and find a good starting point.

7.  **Future Predictions:**
     -  **Purpose:** Predict future values by loading the best parameters from the parameter tuning and train a final model with that configuration.
     -   **Key Components:**
        -   Load the best parameters from CSV using pandas.
        -  Use the `TFTModel` with the best parameters.
        -  Use `predict` to generate the forecasting for the future.
    -   **Design Choices:** The best parameters found during tuning are used to train the final model and generate predictions, to maximize performance.

This structure ensures a clean, understandable, and reproducible process for time series forecasting with the TFT model.

---

## 🤝 Contributing

1. **Fork the Project**
2. **Create your Feature Branch:** `git checkout -b feature/AmazingFeature`
3. **Commit your Changes:** `git commit -m 'Add some AmazingFeature'`
4. **Push to the Branch:** `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

- This project utilizes the fantastic **Darts** library for time series, which provides an intuitive and efficient way to manage and process time series data.
- The **Temporal Fusion Transformer** model, with its ability to handle complex temporal dynamics, has been a core inspiration and technology for this project.
- Google Colab for providing free access to GPUs, which greatly speed up deep learning model training.
- The developers and contributors to all other packages involved in this project.
---
```
