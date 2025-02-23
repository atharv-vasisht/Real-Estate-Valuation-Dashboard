import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

# Load the cleaned Zillow dataset
df = pd.read_csv("/Users/AtharvVasisht/Documents/GitHub/Real Estate Valuation Project/data/zillow_housing_cleaned.csv")

# Extract unique states & metros
states = df["StateName"].dropna().unique()
metro_options = {state: df[df["StateName"] == state]["RegionName"].unique() for state in states}
date_columns = df.columns[5:]  # Assuming first 5 columns are date-based

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Real Estate Market Dashboard"),

    # Dropdown to select State
    dcc.Dropdown(
        id="state-dropdown",
        options=[{"label": state, "value": state} for state in states],
        placeholder="Select a State",
        clearable=True
    ),

    # Dropdown to select Metro (Filtered by State)
    dcc.Dropdown(
        id="metro-dropdown",
        placeholder="Select a Metro",
        clearable=True
    ),

    # Dropdown to select Date
    dcc.Dropdown(
        id="date-dropdown",
        options=[{"label": date, "value": date} for date in date_columns],
        value=date_columns[-1],  # Default to latest date
        clearable=False
    ),

    # Graph to display prices
    dcc.Graph(id="price-chart")
])

# Callback to update Metro dropdown based on selected State
@app.callback(
    Output("metro-dropdown", "options"),
    [Input("state-dropdown", "value")]
)
def update_metro_dropdown(selected_state):
    if selected_state is None:
        return []  # No metros if no state is selected
    return [{"label": metro, "value": metro} for metro in metro_options[selected_state]]

# Callback to update the price graph
@app.callback(
    Output("price-chart", "figure"),
    [Input("state-dropdown", "value"),
     Input("metro-dropdown", "value"),
     Input("date-dropdown", "value")]
)
def update_chart(selected_state, selected_metro, selected_date):
    filtered_df = df.copy()

    # Apply State & Metro Filters
    if selected_state:
        filtered_df = filtered_df[filtered_df["StateName"] == selected_state]
    if selected_metro:
        filtered_df = filtered_df[filtered_df["RegionName"] == selected_metro]

    # Ensure data exists
    if filtered_df.empty:
        return px.bar(title="No data available for the selected filters")

    # Format data for visualization
    filtered_df = filtered_df[['RegionName', 'StateName', selected_date]].copy()
    filtered_df.rename(columns={'RegionName': 'City', 'StateName': 'State', selected_date: 'Median Price'}, inplace=True)

    # Create Bar Chart
    figure = px.bar(filtered_df, x="City", y="Median Price", color="State",
                    title=f"Median Housing Prices ({selected_date})",
                    labels={"City": "Metro Area", "Median Price": "Price ($)"})

    return figure

if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8050)