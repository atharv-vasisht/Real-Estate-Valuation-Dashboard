import os
from flask import Flask, render_template
from dash import Dash

# Initialize Flask app
server = Flask(__name__)

# Import Dash apps (ensure they exist in the correct paths)
from dashboard.app import app as dash1
from dashboard.historical_analysis import app as dash2
from scripts.ml_model import app as dash3

# Attach Dash apps to Flask
for dash_app, url_path in [(dash1, "/app/"), (dash2, "/historical/"), (dash3, "/ml-model/")]:
    dash_app.config.update({"requests_pathname_prefix": url_path})
    dash_app.server = server  # Attach to the Flask instance

# Define Flask route for index.html
@server.route("/")
def index():
    return render_template("index.html")

# Bind Flask to Render’s dynamic port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns a port dynamically
    server.run(host="0.0.0.0", port=port, debug=True)
# The code above is the main entry point for the application. It initializes a Flask server and binds the Dash apps to it. The Flask server also serves the index.html file, which contains the navigation links to the different Dash apps.