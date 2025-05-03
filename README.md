# COCA-COLA-STOCK-PRICE-PREDICTOR
# Coca-Cola Stock Price Prediction

This project predicts Coca-Cola stock prices using historical data and a Random Forest Regressor model. It fetches data using the `yfinance` library, performs some feature engineering, trains the model, and evaluates its performance.

##   Key Features

* **Data Acquisition:** Downloads historical Coca-Cola stock data from Yahoo Finance using the `yfinance` library.
* **Data Exploration:** Displays information about the dataset (data types, missing values) and shows the first/last few rows.
* **Data Visualization:** Plots the historical closing prices of the stock.
* **Feature Engineering:** Extracts date components (Year, Month, Day) and calculates moving averages and volatility as technical indicators.
* **Model Training:**
    * Uses a Random Forest Regressor for predicting the closing price.
    * Splits the data into training and testing sets.
* **Model Evaluation:** Calculates Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to assess prediction accuracy.
* **Future Prediction:** Predicts stock prices for the next 30 days.
* **Visualization:** Plots both historical prices and future predictions.

##   Dependencies

* pandas
* numpy
* matplotlib
* seaborn
* yfinance
* scikit-learn (sklearn)

##   Usage

1.  **Install Libraries:** Make sure you have the required Python libraries installed. You can install them using pip:

    ```bash
    pip install pandas numpy matplotlib seaborn yfinance scikit-learn
    ```

2.  **Run the Notebook:** Execute the Jupyter Notebook "COCA COLA STOCK.ipynb" to:

    * Download the stock data.
    * Preprocess the data and train the model.
    * Evaluate the model.
    * Generate and visualize future predictions.

##   Important Considerations

* **Prediction Accuracy:** Stock market predictions are inherently challenging. The model's accuracy can vary, and past performance is not indicative of future results.
* **Model Limitations:** The Random Forest Regressor is a powerful model, but it's still a statistical model. It may not capture all the complex factors that influence stock prices.
* **Data Range:** The accuracy of predictions depends on the quality and range of historical data.
* **No Financial Advice:** This project is for educational purposes only and should not be considered financial advice.

**Disclaimer:** This code is for informational purposes only. Do not make investment decisions based solely on the predictions from this notebook. Consult with a qualified financial advisor before making any investment decisions.
