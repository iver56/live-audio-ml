import math
import os
import random
import uuid

import numpy as np
from PIL import Image
from pathlib import Path

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from keras.utils import Sequence

from ml.classifier.categories import CATEGORIES, NON_LAUGHTER_CATEGORIES
from ml.classifier.prepare_data import (
    load_wav_file,
    preprocess_audio_chunk,
    FIXED_SOUND_LENGTH,
    NUM_MELS,
)
from ml.settings import AUDIO_EVENT_DATASET_PATH, SAMPLE_RATE
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

        self.augmenter = Compose(
            [
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
                Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            ]
        )

    def __len__(self):
        return math.ceil(len(self.sound_file_paths) / self.batch_size)

    def __getitem__(self, idx):
        # Note: The returned batch does not depend on idx at the moment
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

            spectrogram = preprocess_audio_chunk(
                sound_np, fixed_sound_length=self.fixed_sound_length, num_mels=self.num_mels
            )
            if self.save_augmented_images_to_path:
                # Save the augmented image(vectors) to path
                generated_uuid = uuid.uuid4()
                input_image_pil = Image.fromarray((spectrogram * 255).astype(np.uint8))
                input_image_pil.save(
                    os.path.join(
                        self.save_augmented_images_to_path,
                        "{}__{}_input.png".format(
                            Path(sound_file_path).stem, generated_uuid
                        ),
                    )
                )

            x.append(spectrogram)
            y.append(target)

        x = np.array(x)
        y = np.array(y)

        if self.preprocessing_fn:
            x = self.preprocessing_fn(x)

        return x, y
