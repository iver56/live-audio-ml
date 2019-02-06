import functools
import os

import joblib
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from ml.classifier.categories import CATEGORIES, is_laughter_category
from ml.settings import AUDIO_EVENT_DATASET_PATH, DATA_DIR, SAMPLE_RATE
from ml.utils.filename import get_file_paths

NUM_MELS = 100
FFT_WINDOW_SIZE = 2048
HOP_LENGTH = 512  # AKA stride
FIXED_SOUND_LENGTH = 150  # in "spectrogram" windows


def preprocess_audio_chunk(samples):
    """
    :param samples: numpy array of audio samples between -1 and 1.
    :return:
    """
    assert len(samples) >= FFT_WINDOW_SIZE

    spectrogram = librosa.feature.melspectrogram(
        y=samples,
        sr=SAMPLE_RATE,
        n_mels=NUM_MELS,
        power=1.0,
        n_fft=FFT_WINDOW_SIZE,
        hop_length=HOP_LENGTH,
    )

    # Normalize the sound level and squeeze (compress) the peaks a little
    normalization_value = np.percentile(spectrogram, 90)
    spectrogram = np.tanh(spectrogram / normalization_value)

    # print(np.amin(spectrogram), np.amax(spectrogram))
    # plot_matrix(spectrogram, output_image_path=plot_dir / "{}.png".format(file_path.stem))

    # Transpose the matrix, because we want to use each column as a feature vector
    spectrogram = np.transpose(spectrogram)

    # Apply zero padding if the spectrogram is not large enough to fill the whole space
    # Or crop the spectrogram if it is too large
    vectors = np.zeros(shape=(FIXED_SOUND_LENGTH, NUM_MELS), dtype=np.float32)
    window = spectrogram[:FIXED_SOUND_LENGTH]
    actual_window_length = len(window)  # may be smaller than FIXED_SOUND_LENGTH
    vectors[:actual_window_length] = window

    return vectors


@functools.lru_cache(maxsize=9001)
def load_wav_file(sound_file_path):
    sample_rate, sound_np = wavfile.read(sound_file_path)
    if sample_rate != SAMPLE_RATE:
        raise Exception(
            "Unexpected sample rate {} (expected {})".format(
                sample_rate, SAMPLE_RATE
            )
        )

    sound_np = sound_np / 32767  # ends up roughly between -1 and 1
    return sound_np


if __name__ == "__main__":
    plot_dir = DATA_DIR / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    x_sequences = []
    y_values = []

    for category in tqdm(CATEGORIES):
        file_paths = get_file_paths(AUDIO_EVENT_DATASET_PATH / "train" / category)
        target_value = 1 if is_laughter_category(category) else 0

        for file_path in file_paths:
            sound_np = load_wav_file(file_path)

            vectors = preprocess_audio_chunk(sound_np)

            x_sequences.append(vectors)
            y_values.append(target_value)

    x_sequences = np.array(x_sequences, dtype=np.float32)
    y_values = np.array(y_values, dtype=np.float32)

    os.makedirs(DATA_DIR / "prepared_dataset", exist_ok=True)
    joblib.dump(
        {"x_sequences": x_sequences, "y_values": y_values},
        os.path.join(DATA_DIR / "prepared_dataset", "dataset.pkl"),
        compress=True,
    )
    print("Stored dataset")
