# server/app.py - CORRECTED

import os
import sys
import logging # Import logging
import threading
import queue
import time
import atexit

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

# --- Dynamic Python Path Adjustment ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    print(f"Adding project root to Python path: {project_root}")
    sys.path.insert(0, project_root)
# ---

# --- Import Project Modules ---
try:
    from src.model import model_loader
    from src.model import prediction_handler
    from server import audio_stream
    from server import audio_buffer
    from server import audio_prep
    from server import audio_processor
except ImportError as e:
    print("="*50)
    print(f"Error: Could not import one or more required modules: {e}")
    print("Please ensure all necessary __init__.py files exist (in server, src, src/model).")
    print("="*50)
    sys.exit(1)
# ---

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
# Get the logger instance for this module <<< FIX ADDED HERE
log = logging.getLogger(__name__)

# Initialize Flask app and SocketIO
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'replace_this_with_a_real_secret_key!'
socketio = SocketIO(app, async_mode='eventlet')

# --- Load Model at Startup ---
MODEL_FILENAME = 'updated_model.h5'
MODEL_PATH = os.path.join(project_root, 'models', MODEL_FILENAME)
loaded_model = None
try:
    log.info(f"Attempting to load model from: {MODEL_PATH}") # Use log variable
    if os.path.exists(MODEL_PATH):
        loaded_model = model_loader.load_trained_model(MODEL_PATH)
        if loaded_model:
             log.info("Model loading process completed successfully.") # Use log variable
        else:
             log.error("Model loader returned None without raising an error.") # Use log variable
    else:
        log.error(f"Model file not found at path: {MODEL_PATH}") # Use log variable

except Exception as e:
    log.error(f"An exception occurred during model loading: {e}", exc_info=True) # Use log variable

if loaded_model is None:
    log.warning("------------------------------------------------") # Use log variable
    log.warning("WARNING: Model failed to load. Predictions will not work.") # Use log variable
    log.warning("------------------------------------------------") # Use log variable
# ---

# --- Global variables for background tasks and communication ---
prediction_queue = queue.Queue(maxsize=5)
stop_event = threading.Event()
background_threads = []
# ---

# --- Background Task Definitions ---

def emit_prediction_updates(output_queue: queue.Queue, stop_event: threading.Event, interval_sec: float = 0.1):
    """
    Worker thread function that checks the output queue and emits predictions
    via SocketIO to connected clients.
    """
    log.info("SocketIO emitter task starting.") # Use log variable
    last_sent_tab = None
    while not stop_event.is_set():
        try:
            item = output_queue.get_nowait()
            if item and item.get('type') == 'prediction':
                tab_output = item.get('data')
                # Optional: Only emit if the prediction has changed
                # if tab_output != last_sent_tab:
                #    log.info(f"Emitting prediction update: {tab_output}")
                #    socketio.emit('prediction_update', {'tab': tab_output})
                #    last_sent_tab = tab_output
                socketio.emit('prediction_update', {'tab': tab_output})
        except queue.Empty:
            pass
        except Exception as e:
            log.error(f"Error in emitter task: {e}", exc_info=False) # Use log variable
        socketio.sleep(interval_sec)
    log.info("SocketIO emitter task stopped.") # Use log variable


def start_background_tasks():
    """Initializes and starts all background audio processing threads."""
    global background_threads

    if loaded_model is None:
        log.error("Model not loaded, cannot start background processing.") # Use log variable
        return

    log.info("Starting background tasks...") # Use log variable

    # 1. Start Audio Input Stream
    try:
        log.info("Starting audio stream...") # Use log variable
        # Assuming SAMPLE_RATE is defined in audio_buffer and needed by start_stream
        sample_rate = audio_buffer.SAMPLE_RATE
        audio_stream.start_stream(samplerate=sample_rate)
        log.info("Audio stream started.") # Use log variable
    except Exception as e:
        log.error(f"FATAL: Failed to start audio stream: {e}", exc_info=True) # Use log variable
        return

    # 2. Start Buffer Filling Thread
    try:
        log.info("Starting audio buffer thread...") # Use log variable
        audio_buffer.start_buffer_thread()
        log.info("Audio buffer thread started.") # Use log variable
    except Exception as e:
        log.error(f"FATAL: Failed to start buffer thread: {e}", exc_info=True) # Use log variable
        return

    log.info("Allowing initial buffer fill (2s)...") # Use log variable
    time.sleep(2)

    # 3. Start the Prediction Loop Thread
    log.info("Starting prediction loop thread...") # Use log variable
    try:
        handler_func = prediction_handler.get_tab_output
        pred_thread = threading.Thread(
            target=audio_processor.run_prediction_loop,
            args=(loaded_model, handler_func, prediction_queue, stop_event, audio_buffer.SAMPLE_RATE),
            name="PredictionLoopThread",
            daemon=True
        )
        pred_thread.start()
        background_threads.append(pred_thread)
        log.info("Prediction loop thread started.") # Use log variable
    except Exception as e:
        log.error(f"FATAL: Failed to start prediction loop thread: {e}", exc_info=True) # Use log variable
        return

    # 4. Start the SocketIO Emitter Task
    log.info("Starting SocketIO emitter task...") # Use log variable
    try:
         socketio.start_background_task(target=emit_prediction_updates,
                                        output_queue=prediction_queue,
                                        stop_event=stop_event)
         log.info("SocketIO emitter task started.") # Use log variable
    except Exception as e:
        log.error(f"FATAL: Failed to start SocketIO emitter task: {e}", exc_info=True) # Use log variable
        return

    log.info("All background tasks initiated.") # Use log variable

# --- Web Routes and SocketIO Handlers ---
@app.route('/')
def index():
    # log.info("HTTP Request: Serving index.html") # Use log variable
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    log.info(f"Client connected: {request.sid}") # Use log variable
    emit('prediction_update', {'tab': [0, 0, 0, 0, 0, 0]}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    log.info(f"Client disconnected: {request.sid}") # Use log variable

@socketio.on_error_default
def default_error_handler(e):
    log.error(f"SocketIO Error: {e} (Client SID: {request.sid if request else 'N/A'})") # Use log variable


# --- Graceful Shutdown Logic ---
def shutdown_server():
    """Signals background threads to stop and cleans up."""
    log.info("Shutdown requested. Signaling background tasks...") # Use log variable
    stop_event.set()

    try:
        log.info("Stopping audio buffer thread...") # Use log variable
        # Make sure stop_buffer_thread is implemented correctly in audio_buffer.py
        if hasattr(audio_buffer, 'stop_buffer_thread'):
             audio_buffer.stop_buffer_thread()
        else:
             log.warning("audio_buffer.stop_buffer_thread() function not found.")
    except Exception as e:
        log.error(f"Error stopping audio buffer thread: {e}") # Use log variable

    try:
        log.info("Stopping audio stream...") # Use log variable
        # Add a stop_stream() function to audio_stream.py or use sd.stop()
        if hasattr(audio_stream, 'stop_stream'):
           audio_stream.stop_stream()
           log.info("Called audio_stream.stop_stream()")
        else:
           import sounddevice as sd
           sd.stop()
           log.info("Called sounddevice.stop()")
    except Exception as e:
        log.error(f"Error stopping audio stream: {e}") # Use log variable

    log.info("Shutdown sequence completed.") # Use log variable

# Register the shutdown function
atexit.register(shutdown_server)

# Start background tasks in run.py now
# if __name__ != '__main__':
#     start_background_tasks() # Avoid calling here if using run.py

# --- Main Execution (handled by run.py) ---
if __name__ == '__main__':
     print("Starting Flask app directly (use run.py for SocketIO & background tasks)...")
     # Background tasks won't start automatically if run this way unless called explicitly
     # start_background_tasks()
     socketio.run(app, host='0.0.0.0', port=5001, debug=True, use_reloader=False)