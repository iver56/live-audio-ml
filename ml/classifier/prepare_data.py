import functools

import librosa
import numpy as np
from scipy.io import wavfile

from ml.settings import SAMPLE_RATE

NUM_MELS = 100
FFT_WINDOW_SIZE = 2048
HOP_LENGTH = 512  # AKA stride
FIXED_SOUND_LENGTH = 150  # in "spectrogram" windows


def preprocess_audio_chunk(samples, fixed_sound_length=FIXED_SOUND_LENGTH, num_mels=NUM_MELS):
    """
    :param samples: numpy array of audio samples between -1 and 1.
    :return:
    """
    num_samples = len(samples)
    assert num_samples >= FFT_WINDOW_SIZE

    max_num_samples = fixed_sound_length * HOP_LENGTH + FFT_WINDOW_SIZE
    if num_samples > max_num_samples:
        # If we have more samples than we need, cut off those that we are not going to use.
        # We do this to avoid unneeded computation
        samples = samples[-max_num_samples:]

    spectrogram = librosa.feature.melspectrogram(
        y=samples,
        sr=SAMPLE_RATE,
        n_mels=num_mels,
        power=1.0,
        n_fft=FFT_WINDOW_SIZE,
        hop_length=HOP_LENGTH,
    )

    # Normalize the sound level and squeeze (compress) the peaks a little
    normalization_value = np.percentile(spectrogram, 90)
    spectrogram = np.tanh(spectrogram / (normalization_value + 0.0001))

    # print(np.amin(spectrogram), np.amax(spectrogram))
    # plot_matrix(spectrogram, output_image_path=plot_dir / "{}.png".format(file_path.stem))

    # Transpose the matrix, because we want to use each column as a feature vector
    spectrogram = np.transpose(spectrogram)

    # Apply zero padding if the spectrogram is not large enough to fill the whole space
    # Or crop the spectrogram if it is too large
    vectors = np.zeros(shape=(fixed_sound_length, num_mels), dtype=np.float32)
    window = spectrogram[:fixed_sound_length]
    actual_window_length = len(window)  # may be smaller than FIXED_SOUND_LENGTH
    vectors[-actual_window_length:] = window

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

    if sound_np.dtype != np.float32:
        assert sound_np.dtype == np.int16
        sound_np = sound_np / 32767  # ends up between -1 and 1

    return sound_np
