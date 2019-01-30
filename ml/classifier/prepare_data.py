import os

import librosa
import numpy as np
from scipy.io import wavfile

from ml.settings import AUDIO_EVENT_DATASET_PATH, DATA_DIR
from ml.utils.filename import get_file_paths
from ml.utils.plot import plot_matrix

if __name__ == "__main__":
    plot_dir = DATA_DIR / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    file_paths = get_file_paths(AUDIO_EVENT_DATASET_PATH / "train" / "laughter")
    for file_path in file_paths:
        print(file_path)
        sample_rate, sound_np = wavfile.read(file_path)
        print(sample_rate)
        print(sound_np.shape, sound_np.dtype, np.amin(sound_np), np.amax(sound_np))

        sound_np = sound_np / 32767  # ends up roughly between -1 and 1

        # Normalize the sound level and squeeze (compress) the peaks a little
        normalization_value = np.percentile(np.abs(sound_np), 75)
        sound_np = np.tanh(sound_np / normalization_value)

        spectrogram = librosa.feature.melspectrogram(
            y=sound_np, sr=sample_rate, n_mels=100, power=1.0
        )
        plot_matrix(
            spectrogram, output_image_path=plot_dir / "{}.png".format(file_path.stem)
        )
