import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
from pyngrok import ngrok

# ✅ Load the cleaned Zillow dataset
df = pd.read_csv("/Users/AtharvVasisht/Documents/GitHub/Real Estate Valuation Project/data/zillow_housing_cleaned.csv")

# ✅ Extract unique states & metros
states = df["StateName"].dropna().unique()
metro_options = {state: df[df["StateName"] == state]["RegionName"].unique() for state in states}
date_columns = df.columns[5:]  # Assuming first 5 columns are date-based

# ✅ Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("📊 Historical Housing Market Analysis", style={'text-align': 'center', 'font-size': '32px'}),

    # ✅ Dropdown for selecting state
    dcc.Dropdown(
        id="state-dropdown",
        options=[{"label": state, "value": state} for state in states],
        placeholder="Select a State",
        clearable=True
    ),

    # ✅ Dropdown for selecting metro
    dcc.Dropdown(
        id="metro-dropdown",
        placeholder="Select a Metro",
        clearable=True
    ),

    # ✅ Graph Output
    dcc.Graph(id="historical-chart")
])

# ✅ Callback to update metro dropdown based on state
@app.callback(
    Output("metro-dropdown", "options"),
    [Input("state-dropdown", "value")]
)
def update_metro_dropdown(selected_state):
    if selected_state is None:
        return []
    return [{"label": metro, "value": metro} for metro in metro_options[selected_state]]

# ✅ Callback to update the chart
@app.callback(
    Output("historical-chart", "figure"),
    [Input("state-dropdown", "value"),
     Input("metro-dropdown", "value")]
)
def update_historical_chart(selected_state, selected_metro):
    filtered_df = df.copy()

    if selected_state:
        filtered_df = filtered_df[filtered_df["StateName"] == selected_state]
    if selected_metro:
        filtered_df = filtered_df[filtered_df["RegionName"] == selected_metro]

    if filtered_df.empty:
        return px.line(title="No data available for the selected filters")

    # Create time series data
    time_series_data = []
    for _, row in filtered_df.iterrows():
        for date in date_columns:
            time_series_data.append({
                'Date': date,
                'Price': row[date],
                'City': row['RegionName'],
                'State': row['StateName']
            })

    time_series_df = pd.DataFrame(time_series_data)

    return px.line(time_series_df, 
                  x='Date', 
                  y='Price',
                  color='City',
                  title='Historical Housing Prices Over Time',
                  labels={'Price': 'Median Price ($)'})

if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(8051).public_url
    print(f" * Public URL: {public_url}")
    
    # Run the server
    app.run_server(debug=True, host="127.0.0.1", port=8051)

