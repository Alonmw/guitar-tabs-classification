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

def get_data_dir(pick_flag = False):
    """
        function that returns preprocessed data as np directory.
        flag determines whether to keep f_pick and n_pick as keys.
    """
    data = {}
    key = ''
    values = []
    for dirpath, _, files in os.walk(output_path):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(dirpath, file)
                values.append(np.load(file_path))
        if values:
            if pick_flag:
                key = os.path.basename(dirpath)
            else:
                key = os.path.basename(os.path.dirname(dirpath))
            values = np.expand_dims(values, axis=-1)
            data.setdefault(key, []).extend(values)
            values = []
    return data
if __name__ == '__main__':
    data = get_data_dir()
    print(data['A0'][0].shape)