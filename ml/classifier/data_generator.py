import os
import random

import numpy as np

from ml.classifier.categories import CATEGORIES, NON_LAUGHTER_CATEGORIES
from ml.classifier.prepare_data import load_wav_file, preprocess_audio_chunk
from ml.settings import AUDIO_EVENT_DATASET_PATH
from ml.utils.filename import get_file_paths

LAUGHTER_CLASS_RATIO = 0.3


def sound_example_generator(batch_size=16, augment=True, save_augmented_images_to_path=None):
    sound_file_paths = {}
    for category in CATEGORIES:
        sound_file_paths[category] = get_file_paths(
            AUDIO_EVENT_DATASET_PATH / "train" / category
        )
        assert len(sound_file_paths[category]) > 0

    if save_augmented_images_to_path:
        os.makedirs(save_augmented_images_to_path, exist_ok=True)

    while True:
        x = []
        y = []
        for _ in range(batch_size):

            if random.random() < LAUGHTER_CLASS_RATIO:
                sound_file_path = random.choice(sound_file_paths["laughter"])
                target = 1  # laughter
            else:
                category = random.choice(NON_LAUGHTER_CATEGORIES)
                sound_file_path = random.choice(sound_file_paths[category])
                target = 0  # not laughter

            sound_np = load_wav_file(sound_file_path)

            if augment:
                pass  # TODO: Apply data augmentation

            vectors = preprocess_audio_chunk(sound_np)

            x.append(vectors)
            y.append(target)

        x = np.array(x)
        y = np.array(y)

        yield x, y
