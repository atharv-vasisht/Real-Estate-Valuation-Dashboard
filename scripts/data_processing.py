import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load raw Zillow data
df = pd.read_csv("/Users/AtharvVasisht/Documents/GitHub/Real Estate Valuation Project/data/zillow_housing_data.csv")

# Only analyze the 150 largest metros for accuracy
df = df.iloc[:150]

# Extract only home price columns (ignoring metadata)
price_columns = df.columns[5:]  # Assuming first 5 columns are metadata

def fill_missing_with_arima(series):
    """ Uses ARIMA to predict and fill missing values in a metro's time series. """
    series = series.astype(float)  # Ensure numerical format
    missing_idx = series.isnull()

    # Step 1: Fill missing values at the very beginning with the first known value
    first_valid_index = series.first_valid_index()
    if first_valid_index is not None and isinstance(first_valid_index, int) and first_valid_index > 0:
        series.iloc[:first_valid_index] = series.iloc[first_valid_index]

    # Step 2: Apply ARIMA for remaining missing values (middle/end)
    if missing_idx.sum() > 0:
        known_values = series.dropna().values
        known_indices = np.where(~series.isnull())[0]  # Indices of known values
        missing_indices = np.where(series.isnull())[0]  # Indices of missing values

        # **Skip ARIMA if there aren’t enough data points**
        if len(known_values) < 5 or len(missing_indices) == 0:
            series.interpolate(method='linear', inplace=True)
            return series

        try:
            model = ARIMA(known_values, order=(2, 1, 2), 
                          enforce_stationarity=False, 
                          enforce_invertibility=False)
            model_fit = model.fit()

            # Ensure we don’t predict more steps than available missing values
            forecast_steps = min(len(missing_indices), 10)
            predicted_values = model_fit.forecast(steps=forecast_steps)

            # Correct indexing issue by using missing_indices directly
            for i in range(forecast_steps):
                series.iloc[missing_indices[i]] = predicted_values[i]

        except Exception as e:
            print(f"ARIMA failed for metro area: {e}")
            series.interpolate(method='linear', inplace=True)  # Fallback

    return series

# Apply ARIMA regression to each metro's time-series
df[price_columns] = df[price_columns].apply(fill_missing_with_arima, axis=1)

# Save the cleaned data
df.to_csv("/Users/AtharvVasisht/Documents/GitHub/Real Estate Valuation Project/data/zillow_housing_cleaned.csv", index=False)

print("Missing values filled using ARIMA (or interpolation if needed) and saved to 'zillow_housing_cleaned.csv'.")
