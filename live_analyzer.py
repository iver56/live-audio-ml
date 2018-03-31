import math
import struct
from asciimatics.screen import Screen

import librosa
import numpy as np
import pyaudio

SAMPLES_PER_CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 44100
RECORD_SECONDS = 50
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

BYTES_PER_SAMPLE = p.get_sample_size(FORMAT)
BIT_DEPTH = BYTES_PER_SAMPLE * 8


def stream_audio(screen):
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=SAMPLES_PER_CHUNK
    )

    for i in range(0, int(SAMPLE_RATE / SAMPLES_PER_CHUNK * RECORD_SECONDS)):
        data = stream.read(SAMPLES_PER_CHUNK)

        values = []
        for j in range(SAMPLES_PER_CHUNK):
            byte_index = j * BYTES_PER_SAMPLE
            those_bytes = data[byte_index:byte_index + BYTES_PER_SAMPLE]
            unpacked_int = struct.unpack("<h", those_bytes)[0]
            value = unpacked_int / 32767  # ends up roughly between -1 and 1
            values.append(value)

        values = np.array(values)

        spectrogram = librosa.feature.melspectrogram(
            y=values,
            sr=SAMPLE_RATE,
            n_fft=SAMPLES_PER_CHUNK,
            hop_length=SAMPLES_PER_CHUNK * 2,
            n_mels=100,
            power=1.0
        )
        for mel_band_idx, mel_band_series in enumerate(spectrogram):
            bar_height = int(mel_band_series[0] * screen.height / 2)
            for y in range(screen.height):
                character = '#' if y <= bar_height else ' '
                screen.print_at(
                    character,
                    mel_band_idx,
                    screen.height - y - 1
                )

        screen.refresh()

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == '__main__':
    Screen.wrapper(stream_audio)
