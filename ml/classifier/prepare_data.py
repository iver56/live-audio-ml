import numpy as np
from scipy.io import wavfile

from ml.settings import AUDIO_EVENT_DATASET_PATH
from ml.utils.filename import get_file_paths

if __name__ == "__main__":
    file_paths = get_file_paths(AUDIO_EVENT_DATASET_PATH / 'train' / 'laughter')
    for file_path in file_paths:
        print(file_path)
        rate, sound_np = wavfile.read(file_path)
        print(rate)
        print(sound_np.shape, sound_np.dtype, np.amin(sound_np), np.amax(sound_np))
