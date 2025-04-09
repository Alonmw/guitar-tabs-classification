import queue
import sounddevice as sd
from collections import deque

BUFFER_SIZE = 44100 # 2 seconds of 22050 sr
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata[:, 0])

def start_stream(samplerate=22050, blocksize=2048):
    sd.InputStream(callback=audio_callback,
                   channels=1,
                   samplerate=samplerate,
                   blocksize=blocksize).start()