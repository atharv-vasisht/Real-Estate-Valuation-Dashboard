import os
from flask import Flask, render_template
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from dashboard.market import app as dash1
from dashboard.historical_analysis import app as dash2
from scripts.ml_model import app as dash3

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

@app.route("/")
def index():
    return render_template("index.html")  # Ensure correct template loading

# Configure Dash apps under Flask using DispatcherMiddleware
application = DispatcherMiddleware(
    app.wsgi_app, {
        "/market": dash1.server,
        "/historical": dash2.server,
        "/ml-model": dash3.server,
    }
)

# Expose Flask instance
server = app.server

# Run Flask with Render-compatible dynamic port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use 10000 for local testing
    run_simple("0.0.0.0", port, application, use_reloader=False, use_debugger=False)
