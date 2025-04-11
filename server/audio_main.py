import time
import multiprocessing as mp
from server import audio_stream
from server import audio_buffer
from server import audio_prep
from keras import models
import numpy as np
from src.model.prediction_handler import get_tab_output
from src.visualization import ROOT_DIR

MODEL_NAME = "updated_model.h5"
MODEL_PATH = ROOT_DIR + "/models/" + MODEL_NAME
# --- Main Execution Guard ---
if __name__ == '__main__':
    print("Starting application...")
    print("Loading prediction model...")
    prediction_model = models.load_model(MODEL_PATH)
    # --- Set Multiprocessing Start Method ---
    # 'spawn' is generally safer and more consistent across platforms than 'fork'
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Warning: Could not set start method 'spawn' (possibly already set or unavailable): {e}")


    # --- Create Multiprocessing Queues ---
    # These queues handle communication between main process and worker process
    # Limit size to prevent unbounded memory usage if processing falls behind
    MAX_QUEUE_SIZE = 10
    process_input_queue = mp.Queue(maxsize=MAX_QUEUE_SIZE)
    process_output_queue = mp.Queue(maxsize=MAX_QUEUE_SIZE)


    # --- Start Audio Input Stream (using threading as before) ---
    try:
        audio_stream.start_stream()
        print("Audio stream started.")
    except Exception as e:
        print(f"FATAL: Failed to start audio stream: {e}")
        exit(1)

    # --- Allow Buffers to Fill Slightly ---
    print("Allowing initial buffer fill...")
    time.sleep(3)  # Adjust as needed

    # --- Start Buffer Filling Thread (using threading as before) ---
    try:
        audio_buffer.start_buffer_thread()
        print("Audio buffer filling thread started.")
    except Exception as e:
        print(f"FATAL: Failed to start buffer thread: {e}")
        # Optionally stop audio stream here
        exit(1)


    # --- Start Preprocessing Worker Process ---
    worker_process = None
    try:
        worker_process = mp.Process(
            target=audio_prep.preprocessing_worker_process, # Target the function
            args=(process_input_queue, process_output_queue), # Pass the mp.Queues
            name="PreprocessingWorkerProcess",
            daemon=True # Exits automatically if main process exits
        )
        worker_process.start()
        print(f"Preprocessing worker process [{worker_process.pid}] started.")
    except Exception as e:
         print(f"FATAL: Failed to start preprocessing worker process: {e}")
         # Optionally stop other components here
         exit(1)





    # --- Main Processing Loop ---
    print("Starting main processing loop...")
    try:
        while True:
            # 1. Get the latest audio window from the buffer filler thread
            current_window = audio_buffer.get_current_audio_window()

            # 2. If a full window is available, submit it to the worker process
            if current_window is not None:
                try:
                    # Put the window onto the input queue for the worker process
                    # Use put_nowait to avoid blocking the main loop if queue is full
                    process_input_queue.put_nowait(current_window)
                    print("MainLoop: Submitted window for preprocessing.")
                except mp.queues.Full: # Requires importing queue package as mp.queues
                    print("MainLoop: Warning - Preprocessing input queue full. Skipping window.")
                    # Maybe sleep a bit longer if queue is consistently full
                    time.sleep(0.1)
            # else:
                # Optional: Log if buffer isn't full yet
                # print("MainLoop: Buffer not full. Waiting...")

            # 3. Check for results from the worker process (non-blocking)
            try:
                processed_result = process_output_queue.get_nowait()
                print(f"MainLoop: Received processed result (shape {processed_result.shape}).")
                processed_result = np.expand_dims(processed_result, axis=2)
                processed_result = np.expand_dims(processed_result, axis=0)
                prediction = prediction_model.predict(processed_result)
                prediction = get_tab_output(prediction)

                print(f"Prediction: {prediction}")

            except mp.queues.Empty: # Requires importing queue package as mp.queues
                # No result ready from the worker yet, continue loop
                pass

            # 4. Control loop speed
            # Adjust sleep time based on how quickly you need results vs CPU usage
            time.sleep(0.02) # Example: Check ~20 times per second

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")
    except Exception as e:
         print(f"\nUNEXPECTED ERROR in main loop: {e}")
    finally:
        # --- Graceful Shutdown ---
        print("Initiating shutdown sequence...")

        # 1. Signal worker process to stop by sending None
        if worker_process and worker_process.is_alive():
             print("Signaling preprocessing worker to stop...")
             try:
                  process_input_queue.put(None, timeout=1.0) # Send sentinel
             except mp.queues.Full:
                  print("Warning: Could not put stop signal on input queue (full).")
             except Exception as e:
                  print(f"Error sending stop signal: {e}")

        # 2. Stop the buffer filling thread
        print("Stopping audio buffer thread...")
        audio_buffer.stop_buffer_thread() # Ensure this function exists and works

        # 3. Stop the audio stream (important!)
        print("Stopping audio stream...")
        try:
            # Assuming sounddevice is used, find the stream object or use sd.stop()
            # This might require modification to audio_stream.py to allow stopping
            # For now, we'll assume sd.stop() might work if stream wasn't stored
             import sounddevice as sd
             sd.stop() # Attempt to stop the default stream
             print("Audio stream stopped.")
        except Exception as e:
            print(f"Warning: Could not explicitly stop audio stream: {e}")


        # 4. Wait briefly for worker process to exit (optional, esp. if daemon=True)
        # if worker_process and worker_process.is_alive():
        #     print("Waiting for worker process to exit...")
        #     worker_process.join(timeout=3.0) # Wait max 3 seconds
        #     if worker_process.is_alive():
        #         print("Warning: Worker process did not exit gracefully. Terminating.")
        #         worker_process.terminate() # Force kill

        # 5. Close multiprocessing queues (releases resources)
        print("Closing queues...")
        try:
            process_input_queue.close()
            process_output_queue.close()
            # Use join_thread ONLY if you are sure all items have been processed
            # or queues are empty. Otherwise, it can hang.
            # process_input_queue.join_thread()
            # process_output_queue.join_thread()
        except Exception as e:
            print(f"Error closing queues: {e}")


        print("Shutdown complete.")