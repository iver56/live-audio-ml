import hashlib
import json
import math
import os
import random
import uuid

import joblib
import numpy as np
from PIL import Image
from pathlib import Path

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from keras.utils import Sequence

from ml.classifier.categories import CATEGORIES, NON_LAUGHTER_CATEGORIES
from ml.classifier.prepare_data import (
    load_wav_file,
    preprocess_audio_chunk,
    FIXED_SOUND_LENGTH,
    NUM_MELS,
)
from ml.settings import AUDIO_EVENT_DATASET_PATH, SAMPLE_RATE, DATA_DIR
from ml.utils.filename import get_file_paths

LAUGHTER_CLASS_RATIO = 0.3


def get_train_paths():
    sound_file_paths = {}
    for category in CATEGORIES:
        sound_file_paths[category] = get_file_paths(
            AUDIO_EVENT_DATASET_PATH / "train" / category
        )
        assert len(sound_file_paths[category]) > 0

    return sound_file_paths


def get_validation_paths():
    sound_file_paths = {}
    for category in CATEGORIES:
        sound_file_paths[category] = get_file_paths(
            AUDIO_EVENT_DATASET_PATH / "test", filename_prefix=category
        )
        assert len(sound_file_paths[category]) > 0

    return sound_file_paths


class SoundExampleGenerator(Sequence):
    def __init__(
        self,
        sound_file_paths,
        batch_size=8,
        augment=True,
        save_augmented_images_to_path=None,
        fixed_sound_length=FIXED_SOUND_LENGTH,
        num_mels=NUM_MELS,
        preprocessing_fn=None,
        cache_seed=None,
    ):
        self.sound_file_paths = sound_file_paths
        self.batch_size = batch_size
        self.augment = augment
        self.save_augmented_images_to_path = save_augmented_images_to_path
        self.fixed_sound_length = fixed_sound_length
        self.num_mels = num_mels
        self.preprocessing_fn = preprocessing_fn

        if save_augmented_images_to_path:
            os.makedirs(save_augmented_images_to_path, exist_ok=True)

        self.cache_dir = DATA_DIR / "batch_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.augmenter = Compose(
            [
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            ]
        )

        settings_dict = {
            "sound_file_paths": {
                category_name: [
                    str(path) for path in self.sound_file_paths[category_name]
                ]
                for category_name in self.sound_file_paths
            },
            "batch_size": self.batch_size,
            "augment": self.augment,
            "fixed_sound_length": self.fixed_sound_length,
            "num_mels": self.num_mels,
            "preprocessing_fn_name": self.preprocessing_fn.__name__,
            "cache_seed": cache_seed,
        }
        # We use settings_md5 for caching purposes
        self.settings_md5 = hashlib.md5(
            json.dumps(settings_dict).encode("utf-8")
        ).hexdigest()

        self.counter = 0

    def __len__(self):
        num_sounds = sum(
            len(self.sound_file_paths[category_name])
            for category_name in self.sound_file_paths
        )
        return math.ceil(num_sounds / self.batch_size)

    def __getitem__(self, idx):
        # Note: The returned batch does not depend on idx, but on self.counter

        self.counter += 1

        cache_file_path = self.cache_dir / "{}_{:05d}.pkl".format(
            self.settings_md5, self.counter
        )
        if cache_file_path.exists():
            return joblib.load(cache_file_path)

        x = []
        y = []
        for _ in range(self.batch_size):

            if random.random() < LAUGHTER_CLASS_RATIO:
                sound_file_path = random.choice(self.sound_file_paths["laughter"])
                target = 1  # laughter

            else:
                category = random.choice(NON_LAUGHTER_CATEGORIES)
                sound_file_path = random.choice(self.sound_file_paths[category])
                target = 0  # not laughter

            sound_np = load_wav_file(sound_file_path)

            if self.augment:
                sound_np = self.augmenter(samples=sound_np, sample_rate=SAMPLE_RATE)

            vectors = preprocess_audio_chunk(
                sound_np,
                fixed_sound_length=self.fixed_sound_length,
                num_mels=self.num_mels,
            )
            if self.save_augmented_images_to_path:
                # Save the augmented image(vectors) to path
                generated_uuid = uuid.uuid4()
                input_image_pil = Image.fromarray((vectors * 255).astype(np.uint8))
                input_image_pil.save(
                    os.path.join(
                        self.save_augmented_images_to_path,
                        "{}__{}_input.png".format(
                            Path(sound_file_path).stem, generated_uuid
                        ),
                    )
                )

            x.append(vectors)
            y.append(target)

        x = np.array(x)
        y = np.array(y)

        if self.preprocessing_fn:
            x = self.preprocessing_fn(x)

        return_value = (x, y)

        # Cache returned value
        joblib.dump(return_value, cache_file_path, compress=True)

        return return_value
