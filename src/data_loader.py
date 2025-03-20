"""
Loads all the WAV files data in /data/raw/.
Saves the data in /data/preprocessed/ after preprocessing it to CQT spectrograms
in form of Numpy arrays.
"""
from src.preprocessing import preprocess_file
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = PROJECT_ROOT + "\data\\raw\\"
output_path = PROJECT_ROOT + "\data\\preprocessed\\"

print(data_path)
def load_raw_data():
    for dirpath, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.endswith(".wav"):
                input_file_path = os.path.join(dirpath, filename)
                cqt_normalized, _ = preprocess_file(input_file_path)

                # Handle correct path for the output
                relative_path = os.path.relpath(dirpath, data_path)
                output_folder = output_path + relative_path
                filename = filename.replace(".wav", ".npy")
                output_file_path = output_folder + "\\" + filename

                os.makedirs(output_folder, exist_ok=True)

                np.save(output_file_path, cqt_normalized)

if __name__ == '__main__':
    load_raw_data()