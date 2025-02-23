import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output

# ✅ Load the cleaned Zillow dataset
df = pd.read_csv("/Users/AtharvVasisht/Documents/GitHub/Real Estate Valuation Project/data/zillow_housing_cleaned.csv")

# ✅ Convert column names to string format for date consistency
df.columns = df.columns.map(lambda x: str(x) if "-" in str(x) else x)
date_columns = [col for col in df.columns if "-" in col]

# ✅ Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("📊 Historical Housing Market Analysis", style={'text-align': 'center', 'font-size': '32px'}),

    # ✅ Dropdown for selecting time period
    dcc.Dropdown(
        id="time-period-dropdown",
        options=[
            {"label": "1 Year", "value": 12},
            {"label": "5 Years", "value": 60},
            {"label": "10 Years", "value": 120},
            {"label": "All Time", "value": "all"}
        ],
        value="all",  # Default to all-time
        clearable=False,
        style={'width': '40%', 'margin': 'auto'}
    ),

    # ✅ Radio buttons to choose analysis type
    html.Div([
        dcc.RadioItems(
            id="analysis-type",
            options=[
                {"label": " Top 15 Appreciating Metros", "value": "top"},
                {"label": " Bottom 15 Depreciating Metros", "value": "bottom"},
                {"label": " Search a Specific Metro", "value": "search"}
            ],
            value="top",  # Default to top 15 appreciating metros
            inline=True,
            style={'text-align': 'center', 'margin-top': '10px'}
        )
    ]),

    # ✅ Search bar for specific metro
    dcc.Input(
        id="metro-search",
        type="text",
        placeholder="Enter a Metro (e.g., Seattle, WA)",
        debounce=True,
        style={'display': 'none', 'width': '40%', 'margin': 'auto'}
    ),

    # ✅ Graph Output
    dcc.Graph(id="historical-trends")
])

# ✅ Callback to update search bar visibility
@app.callback(
    Output("metro-search", "style"),
    Input("analysis-type", "value")
)
def toggle_search_visibility(selected_option):
    if selected_option == "search":
        return {'display': 'block', 'width': '40%', 'margin': 'auto'}
    return {'display': 'none'}

# ✅ Callback to update the chart
@app.callback(
    Output("historical-trends", "figure"),
    [Input("time-period-dropdown", "value"),
     Input("analysis-type", "value"),
     Input("metro-search", "value")]
)
def update_chart(period, analysis_type, metro):
    # ✅ Get start and end date columns
    end_date = date_columns[-1]  # Latest date
    if period == "all":
        start_date = date_columns[0]  # First available date
    else:
        start_date_idx = max(0, len(date_columns) - period)
        start_date = date_columns[start_date_idx]

    # ✅ Calculate % change
    df_filtered = df[["RegionName", "StateName", start_date, end_date]].dropna().copy()
    df_filtered["% Change"] = ((df_filtered[end_date] - df_filtered[start_date]) / df_filtered[start_date]) * 100

    # ✅ Format Metro Names
    df_filtered["Metro"] = df_filtered["RegionName"] + ", " + df_filtered["StateName"]

    # ✅ Select metros based on analysis type
    if analysis_type == "top":
        df_filtered = df_filtered.nlargest(15, "% Change")
        title = f"🏆 {period} Housing Market Trends - Top 15 Appreciating Metros"
    elif analysis_type == "bottom":
        df_filtered = df_filtered.nsmallest(15, "% Change")
        title = f"📉 {period} Housing Market Trends - Bottom 15 Depreciating Metros"
    else:
        df_filtered = df_filtered[df_filtered["Metro"].str.contains(metro, case=False, na=False)]
        title = f"📍 {period} Housing Market Trends - {metro}"

    # ✅ Create figure
    fig = go.Figure()

    # ✅ Add "Before" bars
    fig.add_trace(go.Bar(
        x=df_filtered["Metro"],
        y=df_filtered[start_date],
        name="Before",
        marker=dict(color="gray"),
        text=df_filtered[start_date].apply(lambda x: f"${x:,.0f}"),
        textposition="inside"
    ))

    # ✅ Add "After" bars
    fig.add_trace(go.Bar(
        x=df_filtered["Metro"],
        y=df_filtered[end_date],
        name="After",
        marker=dict(color=df_filtered["% Change"], colorscale="RdYlGn"),
        text=df_filtered[end_date].apply(lambda x: f"${x:,.0f}"),
        textposition="inside"
    ))

    # ✅ Layout adjustments
    fig.update_layout(
        title=title,
        xaxis=dict(title="Metropolitan Area", tickangle=-30),
        yaxis=dict(title="Median Home Price ($)", tickformat="$,.0f"),
        barmode="group",  # Group bars to show before vs after
        coloraxis=dict(colorbar_title="% Change"),
        template="plotly_white"
    )

    return fig

if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8070)

