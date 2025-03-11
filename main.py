import os
from flask import Flask, render_template
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from dashboard.app import app as dash1
from dashboard.historical_analysis import app as dash2
from scripts.ml_model import app as dash3

# Initialize Flask app
server = Flask(__name__, template_folder="templates")

@server.route("/")
def index():
    return render_template("index.html")  # Ensuring correct template loading

# Configure Dash apps to run under Flask using DispatcherMiddleware
application = DispatcherMiddleware(
    server.wsgi_app, {
        "/app": dash1.server,
        "/historical": dash2.server,
        "/ml-model": dash3.server,
    }
)

# Run Flask with Render-compatible dynamic port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Ensure Render compatibility
    run_simple("0.0.0.0", port, application, use_reloader=True, use_debugger=True)
