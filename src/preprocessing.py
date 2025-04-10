"""
    takes 2 seconds WAV files from data/raw
    saves preprocessed WAV files in data/preprocessed
"""

import librosa
import numpy as np

def load_audio(file_path, sr=22050):
    """Loads audio file and returns waveform"""
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr

def audio_to_cqt(audio, sr):
    """Converts audio into CQT spectrogram"""
    # Apply HPSS to separate harmonics (clean sound)
    np.save("audio_pre_hpss.npy", audio)
    min_val, max_val = np.min(audio), np.max(audio)
    print(f"Preprocessing: Min/Max JUST before HPSS: {min_val}/{max_val}")
    harmonic, _ = librosa.effects.hpss(audio)
    print("Preprocessing: HPSS applied. compute CQT...")
    # Compute CQT for the harmonic
    cqt = librosa.amplitude_to_db(librosa.cqt(harmonic, sr=sr), ref=np.max)
    return cqt

def normalize_cqt(cqt):
    """Applies standardization and returns normalized CQT spectrogram"""
    mean = np.mean(cqt)
    std = np.std(cqt)
    return (cqt - mean) / std

def preprocess_file(file_path, sr=22050):
    loaded_audio, sr = load_audio(file_path, sr=sr)
    cqt = audio_to_cqt(loaded_audio, sr)
    cqt_normalized = normalize_cqt(cqt)
    return cqt_normalized, sr
