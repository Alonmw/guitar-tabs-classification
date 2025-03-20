import librosa.display
import matplotlib.pyplot as plt
from collections import Counter
import os

def plot_cqt(cqt, sr=22050, title='CQT spectrogram'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(cqt, sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(label="dB")
    plt.title(title)
    plt.show()

def get_class_distribution(data_dir):
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
