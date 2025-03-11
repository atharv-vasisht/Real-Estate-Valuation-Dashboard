import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from scipy.spatial import cKDTree
import numpy as np
import requests

# ✅ Load the cleaned Zillow dataset
df = pd.read_csv("/Users/AtharvVasisht/Documents/GitHub/Real-Estate-Valuation-Project/data/zillow_housing_cleaned.csv")

# ✅ Load correct latitude & longitude data for metros
geo_data = pd.read_csv("https://raw.githubusercontent.com/kelvins/US-Cities-Database/main/csv/us_cities.csv")

# ✅ Rename geo dataset columns to match Zillow format
geo_data.rename(columns={
    "CITY": "RegionName",
    "STATE_CODE": "StateName",
    "LATITUDE": "lat",
    "LONGITUDE": "lon"
}, inplace=True)

# ✅ Standardize geo city names to match Zillow format
geo_data["RegionName"] = geo_data["RegionName"] + ", " + geo_data["StateName"]

# ✅ Ensure unique mapping of city-state pairs
geo_data = geo_data.drop_duplicates(subset=["RegionName", "StateName"])

# ✅ Merge instead of dictionary lookup to prevent mismatches
df = df.merge(geo_data, on=["RegionName", "StateName"], how="left")

# ✅ Convert column names to string format for date consistency
df.columns = df.columns.map(lambda x: str(x) if "-" in str(x) else x)
date_columns = [col for col in df.columns if "-" in col]

# ✅ Fill missing values to ensure no NaN errors
df[date_columns] = df[date_columns].fillna(0)

# ✅ Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("U.S. Housing Market Map"),

    # ✅ Dropdown for selecting date
    dcc.Dropdown(
        id="date-dropdown",
        options=[{"label": date, "value": date} for date in date_columns],
        value=date_columns[0],  # Default to first date
        clearable=False
    ),

    # ✅ Toggle switch to enable/disable animation
    dcc.Checklist(
        id="animation-toggle",
        options=[{"label": " Enable Animation", "value": "animate"}],
        value=[],
        inline=True
    ),

    # ✅ Choropleth Map
    dcc.Graph(id="housing-map")
])

# ✅ Callback to update map based on date & animation toggle
@app.callback(
    Output("housing-map", "figure"),
    [Input("date-dropdown", "value"),
     Input("animation-toggle", "value")]
)
def update_map(selected_date, animation_enabled):
    selected_date = str(selected_date)

    if selected_date not in df.columns:
        return px.scatter_geo(title="Invalid Date Selected")

    # ✅ Filter data for selected date
    df_filtered = df[["RegionName", "StateName", "lat", "lon", selected_date]].dropna().copy()
    df_filtered = df_filtered.rename(columns={selected_date: "Median Home Price"})

    # ✅ Create base figure
    fig = px.scatter_geo(df_filtered,
                         lat="lat", lon="lon",
                         text="RegionName",
                         size="Median Home Price",
                         color="Median Home Price",
                         color_continuous_scale="Plasma",
                         hover_name="RegionName",
                         projection="albers usa",
                         title=f"U.S. Housing Market - {selected_date}")

    # ✅ Enable animation if checkbox is checked
    if "animate" in animation_enabled:
        frames = [
            go.Frame(
                data=[
                    go.Scattergeo(
                        lon=df["lon"],
                        lat=df["lat"],
                        text=df["RegionName"],
                        mode="markers",
                        marker=dict(
                            size=df[date] / 100000,  # Normalize size
                            color=df[date],
                            colorscale="Plasma",
                            showscale=True
                        )
                    )
                ],
                name=date
            )
            for date in date_columns
        ]

        # ✅ Add animation layout
        fig.update_layout(
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Date: "},
                "steps": [
                    {"label": date, "method": "animate", "args": [[date], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}]}
                    for date in date_columns
                ]
            }],
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}], "label": "Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": "Pause", "method": "animate"}
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": True,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": -0.1,
                "yanchor": "top"
            }]
        )

        # ✅ Assign frames to figure
        fig.frames = frames

    return fig


# ✅ Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8060))  # Default 8050 for local testing
    app.run_server(debug=True, host="0.0.0.0", port=port)

'''
if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8080)'
'''
