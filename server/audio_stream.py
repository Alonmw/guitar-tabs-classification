# server/audio_stream.py (PyAudio Version)

import queue
import pyaudio # Import PyAudio
import numpy as np
import time # Keep time for potential sleeps if needed
import sys

# --- Configuration ---
SAMPLE_RATE = 22050
# BLOCK_SIZE equivalent for PyAudio is frames_per_buffer
FRAMES_PER_BUFFER = 2048 # Let's use the blocksize from your start_stream call
CHANNELS = 1
# We'll request 16-bit integer format, common & compatible, then convert to float32
AUDIO_FORMAT = pyaudio.paInt16
NUMPY_FORMAT = np.int16
NORMALIZATION_FACTOR = 32768.0 # For converting int16 to float range -1.0 to 1.0

# Queue for sharing audio data
# Use the maxsize calculated previously if desired
MAX_QUEUE_CHUNKS = int((5 * SAMPLE_RATE) / FRAMES_PER_BUFFER) + 1
audio_queue = queue.Queue(maxsize=MAX_QUEUE_CHUNKS)
print(f"Audio queue initialized with maxsize={MAX_QUEUE_CHUNKS}")


# --- Global variables for PyAudio instance and stream ---
# We need these to manage the stream state (start/stop)
_pyaudio_instance = None
_stream = None
_stop_stream_requested = False # Flag to manage stopping gracefully


# --- PyAudio Callback Function ---
def pyaudio_callback(in_data, frame_count, time_info, status_flags):
    """
    Callback function executed by PyAudio when new audio data is available.
    Converts data to float32 NumPy array and puts it onto the queue.
    """
    global audio_queue
    try:
        # Convert the raw bytes (`in_data`) to a NumPy array of int16
        audio_data_int16 = np.frombuffer(in_data, dtype=NUMPY_FORMAT)

        # Convert int16 array to float32 array (range -1.0 to 1.0)
        audio_data_float32 = audio_data_int16.astype(np.float32) / NORMALIZATION_FACTOR

        # Put the float32 NumPy array onto the queue (non-blocking)
        audio_queue.put_nowait(audio_data_float32)

    except queue.Full:
        # If the queue is full, we drop the data to avoid blocking the callback
        print("Warning: Audio queue full. Discarding audio chunk.", file=sys.stderr) # Requires import sys
        pass # Or implement other handling like logging counts
    except Exception as e:
        # Catch unexpected errors in the callback
        print(f"Error in pyaudio_callback: {e}")
        # Potentially stop the stream or log more details
        return (None, pyaudio.paAbort) # Signal PyAudio to abort stream on error

    # Check stop flag (if implementing graceful stop from callback)
    # if _stop_stream_requested:
    #    return (None, pyaudio.paComplete)

    # Signal PyAudio to continue invoking the callback
    return (None, pyaudio.paContinue)


# --- Stream Management Functions ---
def start_stream(samplerate=SAMPLE_RATE, blocksize=FRAMES_PER_BUFFER):
    """
    Initializes PyAudio and starts the audio input stream.
    """
    global _pyaudio_instance, _stream, FRAMES_PER_BUFFER, SAMPLE_RATE
    global _stop_stream_requested

    if _stream is not None and _stream.is_active():
        print("Stream already running.")
        return

    print(f"Attempting to start PyAudio stream with SR={samplerate}, Blocksize={blocksize}")

    _stop_stream_requested = False
    # Update global constants if different values are passed
    SAMPLE_RATE = samplerate
    FRAMES_PER_BUFFER = blocksize

    try:
        # 1. Initialize PyAudio
        _pyaudio_instance = pyaudio.PyAudio()
        print("PyAudio instance created.")

        # 2. Open the audio stream
        _stream = _pyaudio_instance.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,                   # Specify as input stream
            frames_per_buffer=FRAMES_PER_BUFFER,
            stream_callback=pyaudio_callback # Link the callback function
        )
        print("PyAudio stream opened.")

        # 3. Start the stream callbacks (THIS IS KEY - open() doesn't start it)
        _stream.start_stream()
        print("PyAudio stream started (callbacks active).")

        # Optional: Keep main thread alive if this is the main script
        # while _stream.is_active():
        #    time.sleep(0.1)

    except Exception as e:
        print(f"ERROR opening/starting PyAudio stream: {e}")
        # Clean up if partial initialization occurred
        if _stream is not None:
            _stream.close()
        if _pyaudio_instance is not None:
            _pyaudio_instance.terminate()
        _stream = None
        _pyaudio_instance = None
        raise # Re-raise the exception so the caller knows it failed


def stop_stream():
    """
    Stops and closes the audio stream and terminates PyAudio.
    """
    global _pyaudio_instance, _stream, _stop_stream_requested

    print("Attempting to stop PyAudio stream...")
    _stop_stream_requested = True # Signal callback if needed (optional)

    if _stream is None:
        print("Stream not initialized or already stopped.")
        return

    try:
        if _stream.is_active():
            print("Stopping stream callbacks...")
            _stream.stop_stream()
            print("Stream stopped.")
        else:
             print("Stream was not active.")

        print("Closing stream...")
        _stream.close()
        print("Stream closed.")

    except Exception as e:
        print(f"Error stopping/closing PyAudio stream: {e}")
    finally:
        # Always try to terminate PyAudio instance
        if _pyaudio_instance is not None:
            print("Terminating PyAudio instance...")
            _pyaudio_instance.terminate()
            print("PyAudio instance terminated.")

        # Reset global variables
        _stream = None
        _pyaudio_instance = None
        print("Stream resources released.")

# Example Usage (if running this file directly for testing)
# if __name__ == '__main__':
#     import sys # Needed for printing errors in callback
#     print("Starting audio stream for testing...")
#     start_stream()
#     print("Stream started. Press Ctrl+C to stop.")
#
#     try:
#         # Keep the main thread alive while the stream runs in the background
#         while True:
#             # You can optionally check the queue size here
#             # print(f"Queue size: {audio_queue.qsize()}")
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print("\nKeyboardInterrupt received.")
#     finally:
#         # Ensure the stream is stopped cleanly
#         stop_stream()
#         print("Test finished.")