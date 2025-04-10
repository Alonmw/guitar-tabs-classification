from collections import deque
import numpy as np
from .audio_stream import audio_queue
import queue
import time
from threading import Thread, Lock

SAMPLE_RATE = 22050
WINDOW_SIZE = 2 * SAMPLE_RATE # 2 seconds of recording

# Shared buffer and a lock for thread-safe access
buffer = deque(maxlen=WINDOW_SIZE)
buffer_lock = Lock()
_stop_filling = False # Flag to signal the filling thread to stop
_filler_thread = None

def _fill_buffer_continuously():
    """(Internal) Target function for the background thread."""
    global _stop_filling
    print("Audio buffer filling thread started.")
    while not _stop_filling:
        try:
            # Get audio chunk. block=True waits if queue is empty.
            # Add a timeout to prevent indefinite blocking if stream dies.
            chunk = audio_queue.get(block=True, timeout=1.0) # Wait max 1 sec
            with buffer_lock:
                buffer.extend(chunk) # Add chunk to the deque
        except queue.Empty:
            # Timeout occurred, queue was empty. Continue loop or check stop flag.
            print("Audio queue empty, continuing...") # Optional log
            continue
        except Exception as e:
            print(f"Error in buffer filling thread: {e}")
            # Depending on error, you might want to stop:
            # _stop_filling = True
            time.sleep(0.1) # Avoid busy-looping on repeated errors
    print("Audio buffer filling thread stopped.")

def start_buffer_thread():
    """Starts the background thread to fill the buffer."""
    global _filler_thread, _stop_filling
    if _filler_thread is None or not _filler_thread.is_alive():
        _stop_filling = False
        _filler_thread = Thread(target=_fill_buffer_continuously, daemon=True)
        _filler_thread.start()
        print("Buffer filling thread initiated.")
    else:
        print("Buffer filling thread already running.")

def stop_buffer_thread():
    """Signals the background thread to stop."""
    global _stop_filling
    print("Signalling buffer filling thread to stop...")
    _stop_filling = True
    if _filler_thread and _filler_thread.is_alive():
         _filler_thread.join(timeout=2.0) # Wait briefly for thread to exit
         if _filler_thread.is_alive():
             print("Warning: Buffer filling thread did not stop gracefully.")

def get_current_audio_window():
    """
    Gets a snapshot of the current audio buffer content.
    Returns None if buffer is not yet full.
    """
    with buffer_lock:
        # Only return if the buffer has reached the desired window size
        if len(buffer) == WINDOW_SIZE:
            # Return a *copy* of the buffer contents as a numpy array
            return np.array(buffer)
        else:
            # Return None or an empty array if not full, signalling readiness
            return None

