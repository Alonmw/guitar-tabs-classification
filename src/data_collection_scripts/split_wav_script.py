from pydub import AudioSegment
import os

from src.visualization import ROOT_DIR


def split_wav(file_path, output_dir, segment_length_ms=2000):
    """
    Splits a WAV file into consecutive segments.

    Parameters:
    - file_path: Path to the input WAV file.
    - output_dir: Directory where the segments will be saved.
    - segment_length_ms: Length of each segment in milliseconds (default is 2000ms or 2 seconds).
    """
    # Load the audio file
    audio = AudioSegment.from_wav(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Calculate the number of segments
    total_length_ms = len(audio)
    num_segments = total_length_ms // segment_length_ms

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split and export segments
    for i in range(num_segments):
        start_time = i * segment_length_ms
        end_time = start_time + segment_length_ms
        segment = audio[start_time:end_time]
        segment.export(os.path.join(output_dir, f"{file_name}_part{i+1}.wav"), format="wav")

    # Handle any remaining audio that doesn't fit into a full segment
    if total_length_ms % segment_length_ms != 0:
        start_time = num_segments * segment_length_ms
        segment = audio[start_time:]
        segment.export(os.path.join(output_dir, f"{file_name}_part{num_segments+1}.wav"), format="wav")

    print(f"Splitting complete. Segments saved in '{output_dir}'.")


# input_wav = ROOT_DIR + "" # Relative path to input
# output_directory = "" # Path to output
# split_wav(input_wav, output_directory)
