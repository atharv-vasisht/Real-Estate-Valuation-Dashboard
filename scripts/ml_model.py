import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration of the root logger
)

# ✅ Load the cleaned Zillow dataset
df = pd.read_csv("/Users/AtharvVasisht/Documents/GitHub/Real-Estate-Valuation-Project/data/zillow_housing_cleaned.csv")

# ✅ Convert date columns to datetime format
df.columns = [pd.to_datetime(col, errors='ignore') if '-' in str(col) else col for col in df.columns]
date_columns = [col for col in df.columns if isinstance(col, pd.Timestamp)]

# ✅ Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Housing Market Price Prediction"),

    # Dropdown for selecting metro area
    dcc.Dropdown(
        id="metro-dropdown",
        options=[{"label": region, "value": region} for region in df["RegionName"].unique()],
        value=df["RegionName"].iloc[0],  # Default to first region
        clearable=False
    ),

    # Dropdown for selecting years to predict
    dcc.Dropdown(
        id="years-dropdown",
        options=[{"label": f"{i} Years", "value": i} for i in range(1, 21)],
        value=1,  # Default to 1 year
        clearable=False
    ),

    # Graph to display historical and predicted prices
    dcc.Graph(id="price-graph"),

    # Div to display prediction value
    html.Div(id="prediction-output", style={"margin-top": "20px", "font-size": "18px"})
])

# Train model with improved architecture
model = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2)),  # Reduced to 2 to prevent overfitting
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(score_func=f_regression, k=10)),  # Focus on most important time features
    ("stacked_model", StackingRegressor(
        estimators=[
            ('ridge', Ridge(alpha=0.01)),  # Further reduced alpha
            ('lasso', Lasso(alpha=0.01)),  # Further reduced alpha
            ('elastic', ElasticNet(alpha=0.01, l1_ratio=0.5)),
            ('rf', RandomForestRegressor(
                n_estimators=300,  # Increased trees
                max_depth=8,  # Increased depth
                min_samples_split=2,  # Reduced minimum split
                random_state=42
            ))
        ],
        final_estimator=GradientBoostingRegressor(
            n_estimators=400,  # More trees
            learning_rate=0.01,  # Slower learning
            max_depth=6,  # Deeper trees
            min_samples_split=2,
            subsample=0.9,  # More data used
            random_state=42
        )
    ))
])

# Add trend analysis before prediction
def analyze_trend(y):
    """Analyze the historical trend and adjust predictions accordingly."""
    # Calculate long-term trend
    x = np.arange(len(y))
    trend_model = LinearRegression()
    trend_model.fit(x.reshape(-1, 1), y)
    trend_slope = trend_model.coef_[0]
    
    # Calculate volatility
    returns = np.diff(y) / y[:-1]
    volatility = np.std(returns)
    
    # Calculate market cycle
    window_size = 12  # 1 year
    rolling_mean = pd.Series(y).rolling(window=window_size).mean()
    current_cycle = (y[-1] - rolling_mean.iloc[-1]) / rolling_mean.iloc[-1]
    
    # Adjust trend based on market cycle
    if current_cycle > 0.1:  # Market is above average
        trend_slope *= 1.1  # Increase upward trend
    elif current_cycle < -0.1:  # Market is below average
        trend_slope *= 0.9  # Reduce upward trend
    
    # Calculate historical appreciation rate
    total_years = len(y) / 12
    total_appreciation = (y[-1] - y[0]) / y[0]
    annual_appreciation = (1 + total_appreciation) ** (1/total_years) - 1
    
    # Adjust trend based on historical appreciation
    trend_slope = trend_slope * (1 + annual_appreciation)
    
    return trend_slope, volatility

# Modified prediction function
def make_predictions(model, X, y, future_years):
    """Make predictions with trend adjustment and historical continuity."""
    # Get base predictions
    base_predictions = model.predict(future_years)
    
    # Analyze historical trend
    trend_slope, volatility = analyze_trend(y)
    
    # Get the last known value
    last_known_value = y[-1]
    
    # Calculate trend adjustment with stronger trend influence
    years = np.arange(len(y), len(y) + len(future_years))
    
    # Adjust trend influence based on prediction horizon
    prediction_months = len(future_years)
    if prediction_months <= 12:  # Short-term predictions (1 year or less)
        trend_weight = 1.2  # Reduce trend influence for short-term
        decay_rate = 0.0015  # Slower decay for short-term
    elif prediction_months > 12 and prediction_months <= 24:
        trend_weight = 1.25  # Reduce trend influence for short-term
        decay_rate = 0.0015  # Faster decay for short-term
    elif prediction_months > 24 and prediction_months <= 36:
        trend_weight = 1.3  # Reduce trend influence for short-term
        decay_rate = 0.0015  # Faster decay for short-term
    elif prediction_months > 36 and prediction_months <= 48:
        trend_weight = 1.35  # Reduce trend influence for short-term
        decay_rate = 0.0015  # Faster decay for short-term
    elif prediction_months > 48 and prediction_months <= 60:
        trend_weight = 1.4  # Reduce trend influence for short-term
        decay_rate = 0.0014  # Faster decay for short-term
    elif prediction_months > 60 and prediction_months <= 72:
        trend_weight = 1.45  # Reduce trend influence for short-term
        decay_rate = 0.0014  # Faster decay for short-term
    elif prediction_months > 72 and prediction_months <= 84:
        trend_weight = 1.5  # Reduce trend influence for short-term
        decay_rate = 0.0014  # Faster decay for short-term
    elif prediction_months > 84 and prediction_months <= 96:
        trend_weight = 1.55  # Reduce trend influence for short-term
        decay_rate = 0.0014  # Faster decay for short-term
    elif prediction_months > 96 and prediction_months <= 108:
        trend_weight = 1.6  # Reduce trend influence for short-term
        decay_rate = 0.0014  # Faster decay for short-term
    elif prediction_months > 108 and prediction_months <= 120:
        trend_weight = 1.65  # Reduce trend influence for short-term
        decay_rate = 0.0013  # Faster decay for short-term
    elif prediction_months > 120 and prediction_months <= 132:
        trend_weight = 1.7  # Reduce trend influence for short-term
        decay_rate = 0.0013 # Faster decay for short-term
    elif prediction_months > 132 and prediction_months <= 144:
        trend_weight = 1.75  # Reduce trend influence for short-term
        decay_rate = 0.0013  # Faster decay for short-term
    elif prediction_months > 144 and prediction_months <= 156:
        trend_weight = 1.8  # Reduce trend influence for short-term
        decay_rate = 0.0013  # Faster decay for short-term
    elif prediction_months > 156 and prediction_months <= 168:
        trend_weight = 1.85  # Reduce trend influence for short-term
        decay_rate = 0.0013  # Faster decay for short-term
    elif prediction_months > 168 and prediction_months <= 180:
        trend_weight = 1.9  # Reduce trend influence for short-term
        decay_rate = 0.0012  # Faster decay for short-term
    elif prediction_months > 180 and prediction_months <= 192:
        trend_weight = 1.95  # Reduce trend influence for short-term
        decay_rate = 0.0011  # Faster decay for short-term
    elif prediction_months > 192 and prediction_months <= 204:
        trend_weight = 2  # Reduce trend influence for short-term
        decay_rate = 0.0011  # Faster decay for short-term
    elif prediction_months > 204 and prediction_months <= 216:
        trend_weight = 2.05  # Reduce trend influence for short-term
        decay_rate = 0.0010  # Faster decay for short-term
    elif prediction_months > 216 and prediction_months <= 228:
        trend_weight = 2.1  # Reduce trend influence for short-term
        decay_rate = 0.0010  # Faster decay for short-term
    elif prediction_months > 228 and prediction_months < 240:
        trend_weight = 2.15  # Reduce trend influence for short-term
        decay_rate = 0.0010  # Faster decay for short-term
    else:
        trend_weight = 2.2  # Reduce trend influence for short-term
        decay_rate = 0.0010  # Faster decay for short-term
    
    trend_adjustment = trend_slope * years * np.exp(-decay_rate * years)
    
    # Add scaled randomness based on historical volatility
    random_adjustment = np.random.normal(0, volatility * 0.3, len(future_years))
    
    # Combine adjustments with dynamic weight on trend
    adjusted_predictions = base_predictions + (trend_adjustment * trend_weight) + random_adjustment
    
    # Ensure predictions start from the last known value
    adjusted_predictions[0] = last_known_value
    
    # Allow predictions to deviate more from historical minimum
    min_price = np.min(y) * 0.8  # 20% below historical minimum allowed
    adjusted_predictions = np.maximum(adjusted_predictions, min_price)
    
    # Smooth transitions while preserving the overall trend
    smoothed_predictions = np.copy(adjusted_predictions)
    for i in range(1, len(smoothed_predictions)):
        # Calculate the target value based on the trend
        target_value = adjusted_predictions[i]
        prev_value = smoothed_predictions[i-1]
        
        # Adjust max change based on prediction horizon
        if prediction_months <= 12:
            max_change = volatility * 0.5 * prev_value  # More conservative for short-term
        else:
            max_change = volatility * 1.0 * prev_value  # More flexible for long-term
        
        # If the change is too large, adjust while preserving direction
        if abs(target_value - prev_value) > max_change:
            if target_value > prev_value:
                smoothed_predictions[i] = prev_value + max_change
            else:
                smoothed_predictions[i] = prev_value - max_change
        else:
            smoothed_predictions[i] = target_value
    
    # Ensure the final prediction maintains the overall trend
    final_trend = adjusted_predictions[-1] - adjusted_predictions[0]
    smoothed_trend = smoothed_predictions[-1] - smoothed_predictions[0]
    if abs(final_trend) > abs(smoothed_trend):
        # Adjust the smoothed predictions to maintain the trend
        scale_factor = final_trend / smoothed_trend
        smoothed_predictions = smoothed_predictions[0] + (smoothed_predictions - smoothed_predictions[0]) * scale_factor
    
    return smoothed_predictions

@app.callback(
    [Output("price-graph", "figure"),
     Output("prediction-output", "children")],
    [Input("metro-dropdown", "value"),
     Input("years-dropdown", "value")]
)
def update_graph(selected_metro, years_to_predict):
    try:
        # Filter data for the selected metro
        metro_data = df[df["RegionName"] == selected_metro][["RegionName"] + date_columns].set_index("RegionName").T
        metro_data = metro_data.dropna()

        # Prepare data for training
        X = np.arange(len(metro_data)).reshape(-1, 1)
        y = metro_data.values.flatten()

        # For short-term predictions, use more recent data
        if years_to_predict <= 1:
            # Use last 5 years of data for short-term predictions
            recent_data_size = min(120, len(y))  # 5 years = 60 months
            X = X[-recent_data_size:]
            y = y[-recent_data_size:]
        # Train model
        model.fit(X, y)

        # Make predictions with trend adjustment
        future_years = np.arange(len(X), len(X) + years_to_predict * 12).reshape(-1, 1)
        future_prices = make_predictions(model, X, y, future_years)

        # Append future years to the historical data
        historical_dates = metro_data.index[-len(y):]  # Use dates corresponding to training data
        future_dates = pd.date_range(start=historical_dates[-1], periods=len(future_years) + 1, freq='ME')[1:]

        # Create the figure
        fig = go.Figure()

        # Add historical prices (full history)
        fig.add_trace(go.Scatter(
            x=metro_data.index,
            y=metro_data.values.flatten(),
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue')
        ))

        # Add predicted prices
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices,
            mode='lines+markers',
            name='Predicted Prices',
            line=dict(color='orange', dash='dash')
        ))

        # Add layout with improved formatting
        fig.update_layout(
            title=f"Price Prediction for {selected_metro}",
            xaxis_title="Year",
            yaxis_title="Price (in USD)",
            legend=dict(x=0, y=1.1),
            yaxis=dict(tickformat="$,.0f"),  # Format y-axis with dollar signs
            hovermode='x unified',  # Improved hover information
            showlegend=True
        )

        # Calculate percentage change using the adjusted predictions
        percentage_change = ((future_prices[-1] - y[-1]) / y[-1]) * 100

        # Display the predicted price and percentage change with enhanced UI
        prediction_output = html.Div([
            html.Span(f"In ", style={"font-weight": "normal", "font-size": "25px"}),
            html.B(f"{years_to_predict} years ", style={"font-weight": "bold", "font-size": "25px"}),
            html.Span(f"({future_dates[-1].year}): ", style={"font-weight": "normal", "font-size": "25px"}),
            html.B(f"${future_prices[-1]:,.2f}", style={"color": "green", "font-weight": "bold", "font-size": "30px"}),
            html.Br(),
            html.Span("Percentage Change: ", style={"font-weight": "normal", "font-size": "25px"}),
            html.B(f"{percentage_change:.2f}%", style={"color": "blue" if percentage_change >= 0 else "red", "font-weight": "bold", "font-size": "30px"})
        ], style={"margin-top": "30px", "text-align": "center"})

        return fig, prediction_output
    except Exception as e:
        logging.error(f"Error in update_graph: {str(e)}")
        return go.Figure(), html.Div("Error occurred while generating predictions. Please try again.")

# Run the Dash app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8070))  # Default 8050 for local testing
    app.run_server(debug=True, host="0.0.0.0", port=port)

