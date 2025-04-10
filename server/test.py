import sounddevice as sd
import librosa
import numpy as np
import time
import traceback
from server import audio_stream # To get start_stream()

# --- Configuration ---
CRASHING_ARRAY_PATH = "audio_pre_hpss.npy" # Or .wav
SAMPLE_RATE = 22050

# --- Test ---
print("Starting audio stream...")
try:
    audio_stream.start_stream() # Start the stream
    print("Audio stream started.")
    time.sleep(1) # Let stream fully initialize

    print(f"Loading array from {CRASHING_ARRAY_PATH}...")
    # Load the array (adjust based on how you saved it)
    if CRASHING_ARRAY_PATH.endswith('.npy'):
         loaded_array = np.load(CRASHING_ARRAY_PATH)
    # elif CRASHING_ARRAY_PATH.endswith('.wav'):
    #     loaded_array, _ = librosa.load(CRASHING_ARRAY_PATH, sr=SAMPLE_RATE)
    else:
         raise ValueError("Unknown file type for saved array")

    loaded_array = loaded_array.astype(np.float32) # Ensure float32
    print(f"Array loaded. Shape: {loaded_array.shape}, Dtype: {loaded_array.dtype}")

    # --- Test STFT ONLY ---
    print("Testing librosa.stft() while stream is active...")
    try:
        stft_result = librosa.stft(loaded_array)
        print(f"STFT successful. Result shape: {stft_result.shape}")
        # If STFT works, the issue might be in other parts of HPSS
    except Exception as e:
        print(f"STFT failed with Python exception: {e}")
        traceback.print_exc()
    print("If no crash occurred here, STFT seems okay.")

    # --- Test HPSS again for confirmation ---
    print("\nTesting librosa.effects.hpss() while stream is active...")
    try:
         h_test, p_test = librosa.effects.hpss(loaded_array)
         print(f"HPSS successful.")
    except Exception as e:
         print(f"HPSS failed with Python exception: {e}")
         traceback.print_exc()
    print("If no crash occurred here, something changed!")


finally:
    print("Stopping audio stream...")
    sd.stop() # Stop the stream
    print("Stream stopped.")