import librosa.display
import matplotlib.pyplot as plt
from collections import Counter
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def plot_cqt(cqt, sr=22050, title='CQT spectrogram'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(cqt, sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(label="dB")
    plt.title(title)
    plt.show()

def plot_class_distribution(data_dir):
    class_counts = Counter()

    for base_dir, _, files in os.walk(data_dir):
        label = os.path.basename(base_dir)  # Folder name as label
        class_counts[label] += len([f for f in files if f.endswith(".npy")])

    # Plot class distribution
    plt.figure(figsize=(10, 4))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=45)
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Spectrograms")
    plt.title("Class Distribution")
    plt.show()

def count_data():
    counter = 0
    data_dir = ROOT_DIR + "\data\\preprocessed\\"
    print("For \\data\\preprocessed\\:")
    for base_dir, _, files in os.walk(data_dir):
        for file in files:
            if os.path.isfile(base_dir + '\\' + file):
                counter+=1
        if counter > 0:
            dir_name = os.path.basename(base_dir)
            print(f'{dir_name} contains {counter} files')
        counter = 0


    data_dir = ROOT_DIR + "\data\\raw\\"
    print('\nFor \\data\\raw\\')
    for base_dir, _, files in os.walk(data_dir):
        for file in files:
            if os.path.isfile(base_dir + '\\' + file):
                counter+=1
        if counter > 0:
            dir_name = os.path.basename(base_dir)
            print(f'{dir_name} contains {counter} files')
        counter = 0

count_data()
