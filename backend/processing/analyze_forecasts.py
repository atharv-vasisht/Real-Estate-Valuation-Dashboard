import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the ensemble forecast data
df = pd.read_csv('lstm_metro_forecast_2025_2050.csv')

# Pivot the data to get years as columns (ensemble forecast)
pivoted_df = df.pivot(index='Metro', columns='Year', values='Forecasted_Price')

# Save the pivoted data
pivoted_df.to_csv('forecast_pivoted.csv')

# Print the first few rows for quick review
print('First few rows of the ensemble forecast (pivoted):')
print(pivoted_df.head())

# Create a function to plot individual city forecasts
def plot_city_forecast(city_name, df):
    city_data = df[df['Metro'] == city_name]
    plt.figure(figsize=(12, 6))
    plt.plot(city_data['Year'], city_data['Forecasted_Price'], marker='o')
    plt.title(f'Ensemble Housing Price Forecast for {city_name}')
    plt.xlabel('Year')
    plt.ylabel('Forecasted Price ($)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'forecast_{city_name.replace(",", "").replace(" ", "_")}_ensemble.png')
    plt.close()

# Create a function to plot multiple cities
def plot_multiple_cities(cities, df):
    plt.figure(figsize=(15, 8))
    for city in cities:
        city_data = df[df['Metro'] == city]
        plt.plot(city_data['Year'], city_data['Forecasted_Price'], 
                marker='o', label=city, alpha=0.7)
    
    plt.title('Ensemble Housing Price Forecasts for Selected Cities')
    plt.xlabel('Year')
    plt.ylabel('Forecasted Price ($)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('forecast_multiple_cities_ensemble.png')
    plt.close()

# Get top 5 metros by size rank
top_metros = ['New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Dallas, TX', 'Houston, TX']

# Plot individual cities
for city in top_metros:
    plot_city_forecast(city, df)

# Plot multiple cities together
plot_multiple_cities(top_metros, df)

# Create a heatmap of the pivoted data
plt.figure(figsize=(20, 10))
sns.heatmap(pivoted_df, cmap='YlOrRd', center=pivoted_df.mean().mean())
plt.title('Ensemble Housing Price Forecast Heatmap')
plt.tight_layout()
plt.savefig('forecast_heatmap_ensemble.png')
plt.close()

print("Analysis complete. Check the generated CSV and image files.") 