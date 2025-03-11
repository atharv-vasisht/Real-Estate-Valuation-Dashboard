import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
import os

# Load the cleaned Zillow dataset
df = pd.read_csv("/Users/AtharvVasisht/Documents/GitHub/Real-Estate-Valuation-Project/data/zillow_housing_cleaned.csv")

# Extract unique states & metros
states = df["StateName"].dropna().unique()
metro_options = {state: df[df["StateName"] == state]["RegionName"].unique() for state in states}
date_columns = df.columns[5:]  # Assuming first 5 columns are date-based

# Initialize Dash app
app = dash.Dash(__name__)

# Dash layout
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

# Dash callbacks
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

# Run the Dash app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))  # Default 8050 for local testing
    app.run_server(debug=True, host="0.0.0.0", port=port)
