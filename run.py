# run.py (at project root) - Updated Version

import sys
import os

# --- Add server directory to path ---
server_dir = os.path.join(os.path.dirname(__file__), 'server')
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)
# ---

try:
    # Import the Flask app instance and the SocketIO instance from server/app.py
    from app import app, socketio, start_background_tasks # Import the new function
except ImportError as e:
    print("="*50)
    print("Error: Could not import 'app', 'socketio', or 'start_background_tasks' from server.app.")
    # ... (rest of error message same as before) ...
    print("="*50)
    sys.exit(1)


if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5001      # Make sure this matches the port you want to use

    print("Starting background audio tasks...")
    start_background_tasks() # Call the function to start threads

    print(f"Starting Flask-SocketIO server on http://{host}:{port}")
    print("Access the app via http://localhost:5001 or http://<your-ip-address>:5001")
    print("Press CTRL+C to stop the server.")

    # Use socketio.run()
    # IMPORTANT: use_reloader=False is highly recommended when managing background threads manually like this.
    # Otherwise, Flask's reloader might start your threads twice.
    socketio.run(app, host=host, port=port, debug=True, use_reloader=False)