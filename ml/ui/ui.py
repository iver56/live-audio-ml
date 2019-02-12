import collections
import math
import os
import uuid
import wave

import pyaudio
import pygame

from ml.settings import SAMPLE_RATE, CUSTOM_AUDIO_SET_DATA_PATH_LAUGHTER, CUSTOM_AUDIO_SET_DATA_PATH_NOT_LAUGHTER

os.makedirs(CUSTOM_AUDIO_SET_DATA_PATH_LAUGHTER, exist_ok=True)
os.makedirs(CUSTOM_AUDIO_SET_DATA_PATH_NOT_LAUGHTER, exist_ok=True)

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
                if event.key == pygame.K_1:
                    filepath = CUSTOM_AUDIO_SET_DATA_PATH_LAUGHTER / "{}.wav".format(uuid.uuid4())
                    save_sound_to_file(filepath, samples_ring_buffer)
                    print('Saved file to laughter dir')

                if event.key == pygame.K_0:
                    filepath = CUSTOM_AUDIO_SET_DATA_PATH_NOT_LAUGHTER / "{}.wav".format(uuid.uuid4())
                    save_sound_to_file(filepath, samples_ring_buffer)
                    print('Saved file to NOT laughter dir')


def save_sound_to_file(file_path, samples_ring_buffer):
    waveFile = wave.open(str(file_path), "wb")
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(p.get_sample_size(FORMAT))
    waveFile.setframerate(SAMPLE_RATE)
    waveFile.writeframes(b"".join(samples_ring_buffer))
    waveFile.close()



if __name__ == "__main__":
    main_ui_loop()
