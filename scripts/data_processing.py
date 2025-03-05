import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load raw Zillow data
df = pd.read_csv("/Users/AtharvVasisht/Documents/GitHub/Real Estate Valuation Project/data/zillow_housing_data.csv", na_values=[',,,'])

# Only analyze the 150 largest metros for accuracy
df = df.iloc[:150]

# Extract only home price columns (ignoring metadata)
price_columns = df.columns[5:] # Assuming first 5 columns are metadata

def forward_fill_zillow_data(df):
    """Forward fills missing values in the Zillow housing data."""
    # Get all date columns (columns containing price data)
    date_columns = [col for col in df.columns if isinstance(col, str) and '-' in col]
    # For each row in the dataframe
    for idx in df.index:
        # Get the price series for this row
        price_series = df.loc[idx, date_columns]
        
        # Find the first valid index (first non-null value)
        first_valid_idx = price_series.first_valid_index()
        
        if first_valid_idx is not None:
            # Get the numeric index of the first valid column
            first_valid_col_idx = date_columns.index(first_valid_idx)
            
            # Forward fill all values before the first valid index
            if first_valid_col_idx > 0:
                # Convert first_valid_index (a timestamp) into an integer index
                first_valid_col_idx = date_columns.index(first_valid_idx)  # Find index position
                # Now use the integer index for slicing
                df.loc[idx, date_columns[:first_valid_col_idx]] = price_series[first_valid_idx]

            
            # Forward fill any remaining NaN values using ffill() instead of deprecated fillna
            df.loc[idx, date_columns] = df.loc[idx, date_columns].ffill()
    
    return df

# Process the data
df = forward_fill_zillow_data(df)

# Save the cleaned data
df.to_csv("/Users/AtharvVasisht/Documents/GitHub/Real Estate Valuation Project/data/zillow_housing_cleaned.csv", index=False)

print("Missing values filled using forward-fill and saved to 'zillow_housing_cleaned.csv'.")
