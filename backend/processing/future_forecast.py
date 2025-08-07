import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import datetime
import multiprocessing as mp
from functools import partial
import tensorflow as tf
import gc

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Load the cleaned Zillow dataset
DATA_PATH = "/Users/AtharvVasisht/Documents/GitHub/Real-Estate-Valuation-Project/backend/data/zillow_housing_cleaned.csv"
df = pd.read_csv(DATA_PATH)

# Identify metadata and date columns
metadata_cols = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
date_cols = [col for col in df.columns if '-' in col]

# Hyperparameters
units = 32
activation = 'relu'
optimizer_name = 'adam'
batch_size = 32
epochs = 50
sequence_length = 36
forecast_horizon_months = 25 * 12
validation_split = 0.2

# Ensemble weights
LSTM_WEIGHT = 0.7
LINREG_WEIGHT = 0.3

# Smoothing parameter
def smooth_forecast(prices, smoothing_level=0.2):
    model = ExponentialSmoothing(prices, trend='add', seasonal=None)
    fit = model.fit(optimized=True)
    return fit.fittedvalues

# Calculate the 90th percentile threshold for inflated metros
all_latest_prices = df[date_cols].apply(lambda row: row.dropna().values[-1], axis=1)
price_threshold = np.percentile(all_latest_prices, 90)
national_mean = np.mean(all_latest_prices)

def discount_to_2025_dollars(price, year, base_year=2025, inflation_rate=0.03):
    """Discount a future price to 2025 dollars."""
    years_ahead = year - base_year
    return price / ((1 + inflation_rate) ** years_ahead)

def process_metro(metro_name, df, date_cols, sequence_length, forecast_horizon_months):
    try:
        metro_data = df[df['RegionName'] == metro_name]
        prices = metro_data[date_cols].values.flatten()
        
        if np.count_nonzero(~np.isnan(prices)) < sequence_length:
            return None
            
        # Remove any NaN values
        prices = prices[~np.isnan(prices)]
        
        # Calculate year-over-year growth rate and volatility
        yoy_growth = np.diff(prices) / prices[:-1]
        max_growth_rate = np.percentile(yoy_growth, 95)
        min_growth_rate = np.percentile(yoy_growth, 5)
        volatility = np.std(yoy_growth)
        
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Sequence creation with overlapping windows
        X, y = [], []
        for i in range(len(prices_scaled) - sequence_length):
            X.append(prices_scaled[i:i+sequence_length])
            y.append(prices_scaled[i+sequence_length])
        X = np.array(X)
        y = np.array(y)
        
        if len(X) < 24:
            return None
            
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, shuffle=False)
        
        # Enhanced model architecture
        model = Sequential([
            LSTM(units, activation=activation, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(units//2, activation=activation),
            Dropout(0.2),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Determine if this metro is inflated
        latest_price = prices[-1]
        is_inflated = latest_price > price_threshold
        penalty_factor = 0.7 if is_inflated else 0.85
        
        # LSTM Forecasting with enhanced constraints
        last_sequence = prices_scaled[-sequence_length:].tolist()
        forecast_scaled = []
        for i in range(forecast_horizon_months):
            input_seq = np.array(last_sequence[-sequence_length:]).reshape((1, sequence_length, 1))
            next_pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
            if i > 0:
                prev_price = forecast_scaled[-1]
                # Only penalize the growth component, not the base
                max_allowed_growth = 1 + penalty_factor * max_growth_rate
                min_allowed_growth = 1 + penalty_factor * min_growth_rate
                volatility_factor = 1 + (volatility * 2)
                max_allowed_growth = min(max_allowed_growth, volatility_factor)
                min_allowed_growth = max(min_allowed_growth, 1/volatility_factor)
                next_pred_scaled = min(max(next_pred_scaled, prev_price * min_allowed_growth), 
                                     prev_price * max_allowed_growth)
            forecast_scaled.append(next_pred_scaled)
            last_sequence.append(next_pred_scaled)
        # Denormalize LSTM forecast
        min_p, max_p = scaler.data_min_[0], scaler.data_max_[0]
        forecast_prices_lstm = [x * (max_p - min_p) + min_p for x in forecast_scaled]
        
        # Linear Regression Forecast
        X_lin = np.arange(len(prices)).reshape(-1, 1)
        y_lin = prices
        lin_reg = LinearRegression().fit(X_lin, y_lin)
        future_X = np.arange(len(prices), len(prices) + forecast_horizon_months).reshape(-1, 1)
        lin_pred = lin_reg.predict(future_X)
        
        # Weighted Ensemble
        ensemble_pred = LSTM_WEIGHT * np.array(forecast_prices_lstm) + LINREG_WEIGHT * np.array(lin_pred)

        # Check for valid length
        if len(ensemble_pred) < 3:
            print(f"Skipping {metro_name}: not enough data for smoothing")
            return None

        # Exponential Smoothing
        try:
            smoothed_pred = smooth_forecast(ensemble_pred, smoothing_level=0.6)
        except Exception as e:
            print(f"Smoothing failed for {metro_name}: {e}")
            return None

        if smoothed_pred is None or len(smoothed_pred) != len(ensemble_pred):
            print(f"Skipping {metro_name}: invalid smoothed_pred")
            return None

        # Optional: Apply small reversion to mean for inflated metros
        if is_inflated:
            smoothed_pred = 0.9 * smoothed_pred + 0.1 * national_mean
        
        # Get dates and aggregate to yearly
        last_date = datetime.datetime.strptime(date_cols[-1], "%Y-%m-%d")
        forecast_dates = [last_date + datetime.timedelta(days=30 * (i + 1)) for i in range(forecast_horizon_months)]
        yearly = {}
        for date, price in zip(forecast_dates, smoothed_pred):
            year = date.year
            yearly.setdefault(year, []).append(price)
        results = []
        for year in sorted(yearly.keys()):
            avg_price = np.mean(yearly[year])
            discounted_price = discount_to_2025_dollars(avg_price, year)
            results.append({
                'Metro': metro_name,
                'Year': year,
                'Forecasted_Price': discounted_price
            })
        # Clear memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        return results
    except Exception as e:
        print(f"Error processing {metro_name}: {str(e)}")
        return None

def main():
    all_metros = df[df['RegionType'] == 'msa'].sort_values('SizeRank')
    metro_names = all_metros['RegionName'].unique()
    num_cores = mp.cpu_count() - 1
    print(f"Using {num_cores} CPU cores")
    with mp.Pool(num_cores) as pool:
        process_func = partial(
            process_metro,
            df=df,
            date_cols=date_cols,
            sequence_length=sequence_length,
            forecast_horizon_months=forecast_horizon_months
        )
        results = pool.map(process_func, metro_names)
    forecast_results = []
    for result in results:
        if result is not None:
            forecast_results.extend(result)
    forecast_df = pd.DataFrame(forecast_results)
    forecast_df.to_csv('lstm_metro_forecast_2025_2050.csv', index=False)
    print("Forecasts saved to lstm_metro_forecast_2025_2050.csv")

if __name__ == '__main__':
    main() 