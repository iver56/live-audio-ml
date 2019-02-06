import collections
import wave
import math
import struct

import numpy as np
import pygame
import pyaudio

from ml.classifier.prepare_data import (
    preprocess_audio_chunk,
    HOP_LENGTH,
    FFT_WINDOW_SIZE,
    FIXED_SOUND_LENGTH,
)
from ml.settings import SAMPLE_RATE

p = pyaudio.PyAudio()
WAVE_OUTPUT_FILENAME = "file.wav"
SAMPLES_PER_CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 5000
NUM_SECONDS = 7


BYTES_PER_SAMPLE = p.get_sample_size(FORMAT)
BIT_DEPTH = BYTES_PER_SAMPLE * 8

pygame.display.init()
pygame.display.set_mode([640, 480])

samples_ring_buffer = collections.deque(
    maxlen=int(math.ceil((NUM_SECONDS * SAMPLE_RATE)/SAMPLES_PER_CHUNK))
)

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=SAMPLES_PER_CHUNK,
)


def main_ui_loop():
    while True:
        data = stream.read(SAMPLES_PER_CHUNK)
        samples_ring_buffer.append(data)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    print("Hey, you pressed the key, '0'!")
                    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                    waveFile.setnchannels(CHANNELS)
                    waveFile.setsampwidth(p.get_sample_size(FORMAT))
                    waveFile.setframerate(SAMPLE_RATE)
                    waveFile.writeframes(b''.join(samples_ring_buffer))
                    waveFile.close()
                    print("We have saved a file")

                if event.key == pygame.K_1:
                    print("Doing whatever")

if __name__ == "__main__":
    main_ui_loop()
