from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Allow requests from your React frontend

# Use environment variable for port, default to 5002
PORT = int(os.environ.get('PORT', 5002))

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/zillow_housing_cleaned.csv')
FORECAST_PATH = os.path.join(os.path.dirname(__file__), '../processing/lstm_metro_forecast_2025_2050.csv')

# Load data once at startup
df = pd.read_csv(DATA_PATH)
forecast_df = pd.read_csv(FORECAST_PATH)

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Real Estate API! Use /api/metros, /api/metro/<metro_name>, or /api/forecast/<metro_name>."})

@app.route('/api/metros')
def get_metros():
    metros = df['RegionName'].unique().tolist()
    return jsonify(metros)

@app.route('/api/metro/<metro_name>')
def get_metro_data(metro_name):
    metro_df = df[df['RegionName'] == metro_name]
    if metro_df.empty:
        return jsonify({'error': 'Metro not found'}), 404
    # Only return date columns and values
    date_cols = [col for col in df.columns if '-' in col]
    prices = metro_df[date_cols].iloc[0].to_dict()
    return jsonify(prices)

@app.route('/api/forecast/<metro_name>')
def get_forecast(metro_name):
    metro_forecast = forecast_df[forecast_df['Metro'] == metro_name]
    if metro_forecast.empty:
        return jsonify({'error': 'Metro not found'}), 404
    years = metro_forecast['Year'].tolist()
    prices = metro_forecast['Forecasted_Price'].tolist()
    return jsonify({'metro': metro_name, 'years': years, 'forecasted_prices': prices})

if __name__ == '__main__':
    app.run(debug=False, port=PORT, host='0.0.0.0')