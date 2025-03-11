# The code above is the main entry point for the application. 
# It initializes a Flask server and binds the Dash apps to it. 
# The Flask server also serves the index.html file, which contains the navigation links to the different Dash apps.
import os
from flask import Flask, render_template
from dash import Dash
import dash_html_components as html

# Initialize Flask app
server = Flask(__name__)

# Import Dash apps (Ensure these files exist)
from dashboard.app import app as dash1
from dashboard.historical_analysis import app as dash2
from scripts.ml_model import app as dash3

# Attach Dash apps inside Flask
def setup_dash(dash_app, url_path):
    dash_app.init_app(server, url_base_pathname=url_path)

setup_dash(dash1, "/app/")
setup_dash(dash2, "/historical/")
setup_dash(dash3, "/ml-model/")

# Define main HTML page
@server.route("/")
def index():
    return render_template("index.html")

# Ensure Flask binds to Render’s dynamic port
if __name__ == "__main__":
    app.run_server(debug=True)  # REMOVE host/port settings
