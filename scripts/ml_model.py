import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# ✅ Load the cleaned Zillow dataset
df = pd.read_csv("/Users/AtharvVasisht/Documents/GitHub/Real Estate Valuation Project/data/zillow_housing_cleaned.csv")

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

@app.callback(
    [Output("price-graph", "figure"),
     Output("prediction-output", "children")],
    [Input("metro-dropdown", "value"),
     Input("years-dropdown", "value")]
)
def update_graph(selected_metro, years_to_predict):
    # Filter data for the selected metro
    metro_data = df[df["RegionName"] == selected_metro][["RegionName"] + date_columns].set_index("RegionName").T
    metro_data = metro_data.dropna()  # Remove NaNs

    # Prepare data for training
    X = np.arange(len(metro_data)).reshape(-1, 1)
    y = metro_data.values.flatten()

    # Train model (Polynomial Regression with Ridge Regularization for stability)
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=2)),
        ("ridge_regression", Ridge(alpha=10.0))  # Ridge regularization with alpha for stability
    ])
    model.fit(X, y)

    # Make predictions for the next `years_to_predict` years
    future_years = np.arange(len(metro_data), len(metro_data) + years_to_predict * 12).reshape(-1, 1)
    future_prices = model.predict(future_years)

    # Append future years to the historical data
    historical_dates = metro_data.index
    future_dates = pd.date_range(start=historical_dates[-1], periods=len(future_years) + 1, freq='M')[1:]
    combined_dates = np.concatenate([historical_dates, future_dates])

    # Combine historical and predicted prices
    combined_prices = np.concatenate([y, future_prices])

    # Create the figure
    fig = go.Figure()

    # Add historical prices
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=y,
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

    # Add layout
    fig.update_layout(
        title=f"Price Prediction for {selected_metro}",
        xaxis_title="Year",
        yaxis_title="Price (in USD)",
        legend=dict(x=0, y=1.1)
    )

    # Calculate percentage change
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

if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8060)