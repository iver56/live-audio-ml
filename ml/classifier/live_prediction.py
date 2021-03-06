import collections
import math
import os
import struct

import numpy as np
import pyaudio
from keras.engine.saving import load_model

from ml.classifier.prepare_data import preprocess_audio_chunk, HOP_LENGTH, FFT_WINDOW_SIZE
from ml.classifier.train_mobilenet import fixed_sound_length, num_mels, \
    preprocess_mobilenet_input
from ml.settings import SAMPLE_RATE, DATA_DIR

SAMPLES_PER_CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 5000

p = pyaudio.PyAudio()

BYTES_PER_SAMPLE = p.get_sample_size(FORMAT)
BIT_DEPTH = BYTES_PER_SAMPLE * 8


def stream_audio():
    samples_ring_buffer = collections.deque(
        maxlen=int(math.ceil((fixed_sound_length + FFT_WINDOW_SIZE / HOP_LENGTH) * HOP_LENGTH))
    )

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=SAMPLES_PER_CHUNK,
    )

    for i in range(0, int(SAMPLE_RATE / SAMPLES_PER_CHUNK * RECORD_SECONDS)):
        data = stream.read(SAMPLES_PER_CHUNK)

        for j in range(SAMPLES_PER_CHUNK):
            byte_index = j * BYTES_PER_SAMPLE
            those_bytes = data[byte_index : byte_index + BYTES_PER_SAMPLE]
            unpacked_int = struct.unpack("<h", those_bytes)[0]
            value = unpacked_int / 32767  # ends up roughly between -1 and 1
            samples_ring_buffer.append(value)

        samples = np.array(samples_ring_buffer)

        spectrogram = preprocess_audio_chunk(
            samples, fixed_sound_length=fixed_sound_length, num_mels=num_mels
        )
        # plot_matrix(spectrogram, output_image_path=plot_dir / "{0:<5}.png".format(i))

        x = np.array([spectrogram])
        x = preprocess_mobilenet_input(x)

        y_predicted = float(model.predict(x)[0])
        print("{:.1f} {}".format(y_predicted, '#' * int(20 * y_predicted)))

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    plot_dir = DATA_DIR / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    model = load_model(os.path.join(DATA_DIR / "models", "mobilenet_v2.h5"))

    stream_audio()
