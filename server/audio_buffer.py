from collections import deque
import numpy as np
from .audio_stream import audio_queue

SAMPLE_RATE = 22050
WINDOW_SIZE = 2 * SAMPLE_RATE # 2 seconds of recording

buffer = deque(maxlen=WINDOW_SIZE)

def get_audio_window():
    while len(buffer) < WINDOW_SIZE:
        chunk = audio_queue.get()
        buffer.extend(chunk)

    return np.array(buffer)
