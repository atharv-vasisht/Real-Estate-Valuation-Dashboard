from flask import Flask, render_template
from dash import Dash
import dash_html_components as html

# Initialize Flask app
server = Flask(__name__)

# Import Dash apps (Ensure these files contain valid Dash apps)
from dashboard.app import app as dash1
from dashboard.historical_analysis import app as dash2
from scripts.ml_model import app as dash3

# Attach Dash apps to Flask
for dash_app, url_path in [(dash1, "/app/"), (dash2, "/historical/"), (dash3, "/ml-model/")]:
    dash_app.config.update({"requests_pathname_prefix": url_path})
    dash_app.server = server  # Attach to the same Flask instance

# Define the main route to serve index.html
@server.route("/")
def index():
    return render_template("index.html")

# Run Flask app
if __name__ == "__main__":
    server.run(debug=True)