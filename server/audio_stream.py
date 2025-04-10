import queue
import sounddevice as sd
import numpy as np

audio_queue = queue.Queue()
SAMPLE_RATE = 22050
BLOCK_SIZE = 2048
MAX_QUEUE_SIZE = int((5 * SAMPLE_RATE) / BLOCK_SIZE) + 1 # Add 1 for buffer

def audio_callback(indata, frames, time, status):
    data_copy = np.copy(indata[:, 0])
    audio_queue.put_nowait(data_copy)

def start_stream(samplerate=22050, blocksize=1024):
    sd.InputStream(callback=audio_callback,
                   channels=1,
                   samplerate=samplerate,
                   blocksize=blocksize).start()