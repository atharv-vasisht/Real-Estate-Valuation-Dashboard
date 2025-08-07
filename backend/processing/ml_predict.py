import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the cleaned Zillow dataset
df = pd.read_csv("/Users/AtharvVasisht/Documents/GitHub/Real-Estate-Valuation-Project/backend/data/zillow_housing_cleaned.csv")

# Show first few rows 
print(df.head())
print(df.columns)

# Identify metadata and date columns
metadata_cols = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
date_cols = [col for col in df.columns if '-' in col]

print("Metadata columns:", metadata_cols)
print("Number of date columns:", len(date_cols))
print("First 5 date columns:", date_cols[:5])
print("Last 5 date columns:", date_cols[-5:])

# --- LSTM By Metro: All 150 Metros with Hyperparameter Tuning ---
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
import plotly.express as px

print("\n--- LSTM By Metro: All 150 Metros (12-month sequence, hyperparameter tuning) ---")

# Get all metros by SizeRank
all_metros = df[df['RegionType'] == 'msa'].sort_values('SizeRank')

# Hyperparameters (single set for speed)
units = 32
activation = 'relu'
optimizer_name = 'adam'
batch_size = 16
epochs = 10

sequence_length = 12
final_results = []

for idx, row in all_metros.iterrows():
    metro_name = row['RegionName']
    print(f"\nProcessing: {metro_name}")
    metro_data = df[df['RegionName'] == metro_name]
    prices = metro_data[date_cols].values.flatten()
    if np.count_nonzero(~np.isnan(prices)) < sequence_length:
        print(f"  Skipped (not enough data)")
        continue
    # Normalize
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    # Sequence
    X, y = [], []
    for i in range(len(prices_scaled) - sequence_length):
        X.append(prices_scaled[i:i+sequence_length])
        y.append(prices_scaled[i+sequence_length])
    X = np.array(X)
    y = np.array(y)
    if len(X) < 24:
        print(f"  Skipped (not enough sequences)")
        continue
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Build and train model
    model = Sequential([
        LSTM(units, activation=activation, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    optimizer = Adam() if optimizer_name == 'adam' else RMSprop()
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    min_p, max_p = scaler.data_min_[0], scaler.data_max_[0]
    y_test_denorm = y_test * (max_p - min_p) + min_p
    y_test_pred_denorm = y_test_pred.flatten() * (max_p - min_p) + min_p
    test_rmse = np.sqrt(mean_squared_error(y_test_denorm, y_test_pred_denorm))
    avg_price = np.mean(y_test_denorm)
    percent_rmse = (test_rmse / avg_price) * 100 if avg_price > 0 else np.nan
    final_results.append((metro_name, test_rmse, percent_rmse))
    print(f"  Test RMSE: {test_rmse:,.0f}, Percent RMSE: {percent_rmse:.2f}%")

# Create DataFrame for all metros
results_df = pd.DataFrame(final_results, columns=['Metro', 'RMSE', 'Percent_RMSE'])

# Plot results
fig = px.bar(results_df, x='Metro', y='Percent_RMSE', 
             title='Percent RMSE by Metro (All 150)', 
             color='Percent_RMSE', 
             color_continuous_scale='Viridis', 
             labels={'Percent_RMSE': 'Percent RMSE (%)'})
fig.show()

# Print final results
print("\n--- Final RMSE Summary (All 150 Metros) ---")
for _, row in results_df.iterrows():
    print(f"{row['Metro']:25s}  RMSE: {row['RMSE']:,.0f}  Percent RMSE: {row['Percent_RMSE']:.2f}%")

# Save results to csv
results_df.to_csv('lstm_metro_rmse_results.csv', index=False)
print("Results saved to lstm_metro_rmse_results.csv")

