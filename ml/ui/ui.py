import collections
import math
import os
import uuid
import wave

import pyaudio
import pygame

from ml.settings import SAMPLE_RATE, CUSTOM_AUDIO_SET_DATA_PATH

os.makedirs(CUSTOM_AUDIO_SET_DATA_PATH, exist_ok=True)

p = pyaudio.PyAudio()
SAMPLES_PER_CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
NUM_SECONDS = 7
BYTES_PER_SAMPLE = p.get_sample_size(FORMAT)

pygame.display.init()
pygame.display.set_mode([640, 480])

samples_ring_buffer = collections.deque(
    maxlen=int(math.ceil((NUM_SECONDS * SAMPLE_RATE) / SAMPLES_PER_CHUNK))
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
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    print("Hey, you pressed the key, '0'!")
                    filepath = CUSTOM_AUDIO_SET_DATA_PATH / "{}.wav".format(uuid.uuid4())
                    waveFile = wave.open(str(filepath), "wb")
                    waveFile.setnchannels(CHANNELS)
                    waveFile.setsampwidth(p.get_sample_size(FORMAT))
                    waveFile.setframerate(SAMPLE_RATE)
                    waveFile.writeframes(b"".join(samples_ring_buffer))
                    waveFile.close()
                    print("We have saved a file")

                if event.key == pygame.K_1:
                    print("Doing whatever")


if __name__ == "__main__":
    main_ui_loop()
