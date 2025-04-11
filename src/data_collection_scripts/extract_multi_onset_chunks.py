import soundfile as sf
import numpy as np
import librosa
import argparse
from pathlib import Path
import math

# --- Configuration ---
# Default target duration for chunks
DEFAULT_TARGET_DURATION_MS = 500
# Minimum length for a chunk to be saved (in samples), avoids tiny fragments
MIN_CHUNK_LENGTH_SAMPLES = 100 # Adjust if needed, e.g., 10ms worth of samples
# Librosa onset detection parameters (tune if needed)
ONSET_HOP_LENGTH = 512 # Standard hop length for STFT-based methods
ONSET_BACKTRACK = True # Tries to align onset to preceding energy minimum
# ---

def extract_chunks_from_onsets(input_wav_file: Path,
                               output_base_dir: Path,
                               label_name: str,
                               target_duration_ms: int):
    """
    Detects multiple onsets in a single WAV file and extracts a chunk of
    target_duration_ms starting from each onset. Saves chunks to a labeled
    subdirectory within output_base_dir.

    Args:
        input_wav_file (Path): Path to the single input WAV file containing multiple plays.
        output_base_dir (Path): The base directory where the output label subdirectory
                                will be created.
        label_name (str): The label name to use for the output subdirectory
                          (e.g., "string_E4", "negative").
        target_duration_ms (int): Desired duration of each extracted chunk in milliseconds.
    """
    print(f"Processing file: {input_wav_file}")
    print(f"Target chunk duration: {target_duration_ms} ms")

    # 1. Validate input file path
    if not input_wav_file.is_file():
        print(f"Error: Input file not found at '{input_wav_file}'")
        return

    # 2. Create output directory (including label subdir)
    output_label_dir = output_base_dir / label_name
    try:
        output_label_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: '{output_label_dir}'")
    except OSError as e:
        print(f"Error creating output directory '{output_label_dir}': {e}")
        return

    # 3. Load audio file
    try:
        data, samplerate = sf.read(input_wav_file, dtype='float32')
        print(f"  Loaded audio: Duration={len(data)/samplerate:.2f}s, Sample Rate={samplerate}Hz")
    except Exception as e:
        print(f"Error loading audio file '{input_wav_file}': {e}")
        return

    # 4. Calculate target chunk size in samples
    target_samples = int(samplerate * target_duration_ms / 1000.0)
    if target_samples <= MIN_CHUNK_LENGTH_SAMPLES:
        print(f"Error: Target duration {target_duration_ms}ms is too short (<= {MIN_CHUNK_LENGTH_SAMPLES} samples) at {samplerate}Hz.")
        return

    # 5. Detect all onsets
    print(f"  Detecting onsets...")
    try:
        # units='samples' gives direct sample indices
        # backtrack=True often gives better alignment with perceived onset
        # hop_length influences the temporal resolution of the onset analysis
        onset_samples = librosa.onset.onset_detect(y=data,
                                                   sr=samplerate,
                                                   units='samples',
                                                   hop_length=ONSET_HOP_LENGTH,
                                                   backtrack=ONSET_BACKTRACK)
                                                   # Consider adding pre_max, post_max, pre_avg, post_avg, delta, wait for tuning
        print(f"  Detected {len(onset_samples)} onsets.")
    except Exception as e:
        print(f"Error during onset detection: {e}")
        return

    if len(onset_samples) == 0:
        print("  No onsets detected in the file.")
        return

    # 6. Extract and save chunk for each onset
    chunks_saved = 0
    input_stem = input_wav_file.stem # Original filename without extension
    pad_width = len(str(len(onset_samples))) # For zero-padding filenames

    for i, onset_sample in enumerate(onset_samples):
        start_sample = onset_sample
        end_sample = start_sample + target_samples

        # Ensure we don't try to start slicing past the end of the audio
        if start_sample >= len(data):
            # print(f"  Onset {i+1} at sample {start_sample} is beyond audio length {len(data)}. Skipping.")
            continue

        # Extract chunk (NumPy handles slicing beyond the end gracefully)
        chunk_data = data[start_sample:end_sample]

        # Check if the extracted chunk is reasonably long
        if chunk_data.size >= MIN_CHUNK_LENGTH_SAMPLES:
            # Construct new filename
            chunk_filename = f"{input_stem}_onset_{str(i+1).zfill(pad_width)}_chunk.wav"
            output_chunk_path = output_label_dir / chunk_filename

            # Save the chunk
            try:
                sf.write(output_chunk_path, chunk_data, samplerate)
                chunks_saved += 1
            except Exception as e:
                print(f"  Error writing chunk file {chunk_filename}: {e}")
        # else:
            # print(f"  Skipping onset {i+1}: Resulting chunk too short ({chunk_data.size} samples).")


    print(f"\nFinished processing {input_wav_file.name}.")
    print(f"Detected {len(onset_samples)} onsets.")
    print(f"Successfully saved {chunks_saved} chunks to '{output_label_dir}'.")
    print(f"--------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect multiple onsets in a WAV file and extract fixed-duration chunks."
    )
    parser.add_argument("input_wav", type=str,
                        help="Path to the single input WAV file containing multiple events.")
    parser.add_argument("output_dir", type=str,
                        help="Base directory to save the output chunks (a label subdir will be created).")
    parser.add_argument("label_name", type=str,
                        help="Label name for the output subdirectory (e.g., 'string_E4').")
    parser.add_argument("--duration_ms", type=int, default=DEFAULT_TARGET_DURATION_MS,
                        help=f"Target duration of each chunk in milliseconds (default: {DEFAULT_TARGET_DURATION_MS}).")

    args = parser.parse_args()

    input_path = Path(args.input_wav)
    output_path = Path(args.output_dir)

    extract_chunks_from_onsets(input_path, output_path, args.label_name, args.duration_ms)