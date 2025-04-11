"""
Preprocesses all WAV files in the provided input directory.
For each file, it computes the normalized CQT spectrogram, expands its dimensions, 
and saves it as a NumPy array in the corresponding output directory, 
preserving the directory structure.
"""
from src.data_utils.preprocessing import preprocess_file
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_data_path = PROJECT_ROOT + "/data/raw/"
_output_path = PROJECT_ROOT + "/data/preprocessed/"
NEGATIVE_CLASS = "negatives"


def preprocess_and_save_wav_files(data_path, output_path):
    """
    Processes all WAV files in the provided input directory to extract the normalized
    Constant-Q Transform (CQT) spectrogram, expands dimensions to ensure the
    result conforms to shape (84, 87, 1), and saves the processed data as NumPy arrays
    in the corresponding output directory while preserving the directory structure.

    Args:
        data_path (str): Path to the input directory containing WAV files.
        output_path (str): Path to the directory where processed NumPy files will be saved.

    Returns:
        None
    """

    for dirpath, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.endswith(".wav"):
                input_file_path = os.path.join(dirpath, filename)
                cqt_normalized, _ = preprocess_file(input_file_path)

                # Handle correct path for the output
                relative_path = os.path.relpath(dirpath, data_path)
                output_folder = output_path + relative_path
                filename = filename.replace(".wav", ".npy")
                output_file_path = output_folder + "/" + filename

                os.makedirs(output_folder, exist_ok=True)
                # Fix saved file shape to (84,87,1)
                np.save(output_file_path, np.expand_dims(cqt_normalized, axis=-1))


def get_data_dir(data_path, pick_flag = False):
    """
    Retrieves preprocessed data stored as NumPy arrays from the specified directory.
    Groups the data by subdirectory names as dictionary keys and organizes
    the data into a NumPy array. If `pick_flag` is True, uses the innermost
    directory name; otherwise, uses the parent directory name as the key.
    
    Args:
        data_path (str): Path to the directory containing preprocessed .npy files.
        pick_flag (bool): Determines whether to use innermost or parent directory
                          name for grouping data.
    
    Returns:
        dict: A dictionary with keys as directory names and values as lists of
              NumPy arrays.
    """
    data = {}
    key = ''
    values = []

    for dirpath, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(dirpath, file)
                values.append(np.load(file_path))
        if values:
            if os.path.basename(dirpath) == NEGATIVE_CLASS:
                key = NEGATIVE_CLASS
            elif pick_flag:
                key = os.path.basename(dirpath)
            else:
                key = os.path.basename(os.path.dirname(dirpath))
            data.setdefault(key, []).extend(values)
            values = []
    return data

def get_xy(data_dir):
    X = []
    y = []
    for key, values in data_dir.items():
        for value in values:
            X.append(value)
            y.append(key)
    return (
        np.array(X),
        LabelEncoder().fit_transform(y)
    )

if __name__ == '__main__':
    pass
