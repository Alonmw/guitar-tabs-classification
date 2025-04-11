# server/audio_processor.py

import time
import numpy as np
import queue # Use standard queue for thread communication
import logging

# Assuming these modules exist in the 'server' directory or path is adjusted
try:
    from server import audio_buffer
    from server import audio_prep
except ImportError:
    # Allow importing if run directly for testing, assuming siblings
    import audio_buffer
    import audio_prep

# Configure logging for this module
log = logging.getLogger(__name__)

def run_prediction_loop(model,
                        prediction_handler_func,
                        output_queue: queue.Queue,
                        stop_event,
                        sample_rate: int,
                        process_interval_sec: float = 0.05):
    """
    Continuously gets audio windows, preprocesses, predicts, handles prediction,
    and puts the result onto the output queue. Runs until stop_event is set.

    Args:
        model: The loaded Keras/TF model object.
        prediction_handler_func: The function to call to convert softmax to tab output
                                 (e.g., prediction_handler.get_tab_output).
        output_queue (queue.Queue): Thread-safe queue to put the resulting tab_output list into.
        stop_event (threading.Event): Event to signal when the loop should stop.
        sample_rate (int): The sample rate required for preprocessing.
        process_interval_sec (float): How often to fetch/process audio (controls loop speed).
    """
    log.info("Audio processing loop starting.")
    last_prediction = None # Keep track to potentially only send changes

    while not stop_event.is_set():
        start_time = time.monotonic()

        # 1. Get Audio Window
        current_window = audio_buffer.get_current_audio_window()

        tab_output = None # Default to no output for this cycle

        if current_window is not None:
            # 2. Preprocess Audio (Directly in this thread)
            processed_data = audio_prep.preprocess_buffer(current_window, sample_rate)

            if processed_data is not None:
                try:
                    # 3. Reshape for Model Prediction (Typical for Keras CNN: Batch, Height, Width, Channels)
                    # Adjust shape based on your specific model's input requirements!
                    # Example assumes CQT output shape (num_bins, num_time_steps)
                    if processed_data.ndim == 2: # Add batch and channel dims
                        processed_reshaped = np.expand_dims(processed_data, axis=0) # Add batch dim -> (1, H, W)
                        processed_reshaped = np.expand_dims(processed_reshaped, axis=-1) # Add channel dim -> (1, H, W, 1)
                    else:
                        log.warning(f"Unexpected preprocessed data shape: {processed_data.shape}")
                        processed_reshaped = None

                    if processed_reshaped is not None:
                        # 4. Predict
                        softmax_output = model.predict(processed_reshaped)

                        # 5. Handle Prediction (Convert to tab format)
                        tab_output = prediction_handler_func(softmax_output)

                except Exception as e:
                    log.error(f"Error during prediction or handling: {e}", exc_info=False) # Set exc_info=True for full traceback
                    tab_output = None # Ensure reset on error

        # 6. Communicate Result (if valid and maybe changed)
        if tab_output is not None: # and tab_output != last_prediction: # Optional: Only send changes
            try:
                # Put the latest valid result in the queue
                # If queue is full, discard older items first before putting
                while output_queue.full():
                    try:
                        output_queue.get_nowait()
                        log.warning("Output queue was full, discarded oldest prediction.")
                    except queue.Empty:
                        break # Should not happen if .full() was true, but safety first
                output_queue.put_nowait({'type': 'prediction', 'data': tab_output})
                last_prediction = tab_output # Update last sent prediction
            except queue.Full:
                # This case should be rare now, but log if it happens
                 log.warning("Output queue still full after attempting to clear. Prediction dropped.")
            except Exception as e:
                log.error(f"Error putting prediction onto output queue: {e}")

        # 7. Control Loop Speed
        processing_time = time.monotonic() - start_time
        sleep_time = max(0, process_interval_sec - processing_time)
        # Use event.wait for sleeping - allows faster exit if stop_event is set
        stop_event.wait(timeout=sleep_time)

    log.info("Audio processing loop stopped.")