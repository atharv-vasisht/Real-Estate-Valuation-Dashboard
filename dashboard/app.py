import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
from flask import jsonify
import json

# Load the cleaned Zillow dataset
df = pd.read_csv("/Users/AtharvVasisht/Documents/GitHub/Real Estate Valuation Project/data/zillow_housing_cleaned.csv")

# Extract unique states & metros
states = df["StateName"].dropna().unique()
metro_options = {state: df[df["StateName"] == state]["RegionName"].unique() for state in states}
date_columns = df.columns[5:]  # Assuming first 5 columns are date-based

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# API Routes
@server.route('/api/states')
def get_states():
    return jsonify({'states': list(states)})

@server.route('/api/metros')
def get_metros():
    state = request.args.get('state')
    if state and state in metro_options:
        return jsonify({'metros': list(metro_options[state])})
    return jsonify({'metros': []})

@server.route('/api/dates')
def get_dates():
    return jsonify({'dates': list(date_columns)})

@server.route('/api/chart-data')
def get_chart_data():
    state = request.args.get('state')
    metro = request.args.get('metro')
    date = request.args.get('date')

    filtered_df = df.copy()

    # Apply State & Metro Filters
    if state:
        filtered_df = filtered_df[filtered_df["StateName"] == state]
    if metro:
        filtered_df = filtered_df[filtered_df["RegionName"] == metro]

    # Ensure data exists
    if filtered_df.empty:
        return jsonify({'error': 'No data available for the selected filters'})

    # Format data for visualization
    filtered_df = filtered_df[['RegionName', 'StateName', date]].copy()
    filtered_df.rename(columns={'RegionName': 'City', 'StateName': 'State', date: 'Median Price'}, inplace=True)

    # Create Bar Chart
    figure = px.bar(filtered_df, x="City", y="Median Price", color="State",
                    title=f"Median Housing Prices ({date})",
                    labels={"City": "Metro Area", "Median Price": "Price ($)"})

    return jsonify({
        'data': figure.to_dict()['data'],
        'layout': figure.to_dict()['layout']
    })

# Dash layout (can be removed if you're only using the API)
app.layout = html.Div([
    html.H1("Real Estate Market Dashboard"),
    dcc.Dropdown(
        id="state-dropdown",
        options=[{"label": state, "value": state} for state in states],
        placeholder="Select a State",
        clearable=True
    ),
    dcc.Dropdown(
        id="metro-dropdown",
        placeholder="Select a Metro",
        clearable=True
    ),
    dcc.Dropdown(
        id="date-dropdown",
        options=[{"label": date, "value": date} for date in date_columns],
        value=date_columns[-1],
        clearable=False
    ),
    dcc.Graph(id="price-chart")
])

# Dash callbacks (can be removed if you're only using the API)
@app.callback(
    Output("metro-dropdown", "options"),
    [Input("state-dropdown", "value")]
)
def update_metro_dropdown(selected_state):
    if selected_state is None:
        return []
    return [{"label": metro, "value": metro} for metro in metro_options[selected_state]]

@app.callback(
    Output("price-chart", "figure"),
    [Input("state-dropdown", "value"),
     Input("metro-dropdown", "value"),
     Input("date-dropdown", "value")]
)
def update_chart(selected_state, selected_metro, selected_date):
    filtered_df = df.copy()

    if selected_state:
        filtered_df = filtered_df[filtered_df["StateName"] == selected_state]
    if selected_metro:
        filtered_df = filtered_df[filtered_df["RegionName"] == selected_metro]

    if filtered_df.empty:
        return px.bar(title="No data available for the selected filters")

    filtered_df = filtered_df[['RegionName', 'StateName', selected_date]].copy()
    filtered_df.rename(columns={'RegionName': 'City', 'StateName': 'State', selected_date: 'Median Price'}, inplace=True)

    return px.bar(filtered_df, x="City", y="Median Price", color="State",
                  title=f"Median Housing Prices ({selected_date})",
                  labels={"City": "Metro Area", "Median Price": "Price ($)"})

if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8050)