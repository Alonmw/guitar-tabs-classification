# server/audio_prep.py

import numpy as np
import librosa
import time
import traceback
import multiprocessing as mp  # Import multiprocessing
from queue import Empty as QueueEmpty # To handle potential queue timeouts if needed later

# --- Import SAMPLE_RATE ---
# Assuming SAMPLE_RATE is defined consistently, e.g., in audio_buffer
# Adjust import path as needed
try:
    from .audio_buffer import SAMPLE_RATE
except ImportError:
    # Fallback or define it explicitly if necessary
    SAMPLE_RATE = 22050
    print("Warning: Using default SAMPLE_RATE in audio_prep.py")


# --- Existing Preprocessing Functions ---

# Assuming audio_to_cqt and normalize_cqt are defined here or imported
# from src.preprocessing import audio_to_cqt, normalize_cqt ...
# (Make sure these imports work within the multiprocessing context if needed)
# For simplicity, let's assume they are correctly defined/imported

def audio_to_cqt(audio, sr):
    # ... (Your implementation from before) ...
    try:
        print("Preprocessing: Applying HPSS...")
        if not np.issubdtype(audio.dtype, np.floating):
             audio = audio.astype(np.float32)
        harmonic, _ = librosa.effects.hpss(audio)
        print("Preprocessing: HPSS complete.")
        print("Preprocessing: Computing CQT...")
        cqt = librosa.cqt(harmonic, sr=sr)
        print("Preprocessing: CQT computed.")
        print("Preprocessing: Converting CQT to dB...")
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        print("Preprocessing: CQT dB conversion complete.")
        return cqt_db
    except Exception as e:
        print(f"Preprocessing: ERROR inside audio_to_cqt: {e}")
        traceback.print_exc()
        return None

def normalize_cqt(cqt):
    # ... (Your implementation from before with std dev check) ...
    try:
        print("Preprocessing: Normalizing CQT...")
        mean = np.mean(cqt)
        std = np.std(cqt)
        print(f"Preprocessing: CQT Mean={mean}, StdDev={std}")
        if std < 1e-8:
            print("Preprocessing: Warning - CQT std dev near zero. Returning zeros.")
            return np.zeros_like(cqt)
        cqt_normalized = (cqt - mean) / std
        print("Preprocessing: Normalization complete.")
        return cqt_normalized
    except Exception as e:
        print(f"Preprocessing: ERROR inside normalize_cqt: {e}")
        traceback.print_exc()
        return None

def preprocess_buffer(audio_buffer: np.ndarray, sample_rate: int):
    """
    (Existing function) Preprocesses an in-memory audio buffer.
    """
    # ... (Your existing validation and logic calling audio_to_cqt, normalize_cqt) ...
    print(f"Audio Prep: Received buffer shape: {audio_buffer.shape}, dtype: {audio_buffer.dtype}")
    if not isinstance(audio_buffer, np.ndarray) or audio_buffer.ndim != 1:
         print(f"Error: Invalid buffer dimensions: {audio_buffer.ndim}. Expected 1.")
         return None
    if audio_buffer.size == 0:
         print("Error: Empty audio buffer.")
         return None

    try:
         audio_buffer_float = audio_buffer.astype(np.float32)
         print(f"Audio Prep: Converted buffer to dtype: {audio_buffer_float.dtype}")
    except Exception as e:
         print(f"Audio Prep: ERROR during float conversion: {e}")
         return None

    try:
        print("Audio Prep: Calling audio_to_cqt...")
        cqt = audio_to_cqt(audio_buffer_float, sr=sample_rate)
        print("Audio Prep: audio_to_cqt call finished.")
        if cqt is None: return None # Propagate failure

        print("Audio Prep: Calling normalize_cqt...")
        cqt_normalized = normalize_cqt(cqt)
        print("Audio Prep: normalize_cqt call finished.")
        if cqt_normalized is None: return None # Propagate failure

        return cqt_normalized

    except Exception as e:
        print(f"Audio Prep: UNCAUGHT EXCEPTION during audio preprocessing: {e}")
        traceback.print_exc()
        return None


# --- New Worker Function for Multiprocessing ---

def preprocessing_worker_process(input_queue: mp.Queue, output_queue: mp.Queue):
    """
    Worker function to run in a separate process.
    Gets audio buffers from input_queue, processes them using preprocess_buffer,
    and puts results onto output_queue.
    """
    pid = mp.current_process().pid
    print(f"Preprocessing worker process [{pid}] started.")

    while True:
        try:
            # Get the next audio window (NumPy array) from the queue
            # This blocks until an item is available
            window_to_process = input_queue.get() # Removed timeout for simplicity, add back if needed

            if window_to_process is None: # Check for sentinel value to stop
                print(f"Worker [{pid}] received stop signal. Exiting.")
                break

            print(f"Worker [{pid}]: Processing received window (shape {window_to_process.shape})...")

            # Perform the actual preprocessing using the existing function
            processed_data = preprocess_buffer(window_to_process, SAMPLE_RATE)

            if processed_data is not None:
                print(f"Worker [{pid}]: Preprocessing successful (output shape {processed_data.shape}).")
                # Send the result back to the main process
                try:
                    # Use put_nowait or put with timeout if output queue might fill up
                    output_queue.put(processed_data)
                except mp.queues.Full: # Requires importing queue package as mp.queues
                     print(f"Worker [{pid}]: Warning - Output queue full. Result discarded.")
                     # Handle appropriately - maybe wait or log more severely
            else:
                # preprocess_buffer returned None (it failed and logged internally)
                print(f"Worker [{pid}]: Preprocessing failed (returned None).")
                # Optionally put a specific "failure" marker on output_queue if needed

        except (KeyboardInterrupt, SystemExit):
             print(f"Worker [{pid}]: Received interrupt. Exiting.")
             break
        except Exception as e:
            # Catch any unexpected errors within the worker loop
            print(f"ERROR in preprocessing_worker_process [{pid}]: {e}")
            traceback.print_exc()
            # Avoid continuous high-CPU loop on repeated errors
            time.sleep(0.1)

    print(f"Preprocessing worker process [{pid}] stopped.")


# --- Functions to manage the worker process ---
# It's often cleaner to manage process start/stop from the main script,
# but you could have helper functions here if preferred. Example:

# _worker_process_instance = None
#
# def start_preprocessing_process(input_q, output_q):
#     global _worker_process_instance
#     if _worker_process_instance is None or not _worker_process_instance.is_alive():
#         _worker_process_instance = mp.Process(
#             target=preprocessing_worker_process,
#             args=(input_q, output_q),
#             name="PreprocessingWorkerProcess",
#             daemon=True # Set daemon=True if you want it to exit automatically with main
#         )
#         _worker_process_instance.start()
#         print("Preprocessing worker process started.")
#     else:
#         print("Preprocessing worker process already running.")
#     return _worker_process_instance
#
# def stop_preprocessing_process(input_q):
#      if _worker_process_instance and _worker_process_instance.is_alive():
#            print("Stopping preprocessing worker process...")
#            try:
#                 # Send sentinel value
#                 input_q.put(None, timeout=1.0)
#                 # Wait for process to finish (optional)
#                 # _worker_process_instance.join(timeout=5.0)
#                 # if _worker_process_instance.is_alive():
#                 #    print("Warning: Worker process did not join gracefully. Terminating.")
#                 #    _worker_process_instance.terminate()
#            except Full: # Import queue.Full
#                 print("Warning: Could not send stop signal, queue full.")
#            except Exception as e:
#                 print(f"Error stopping worker: {e}")
#      else:
#            print("Worker process not running or already stopped.")