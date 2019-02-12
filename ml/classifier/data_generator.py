import os
import random
import uuid

import numpy as np
from PIL import Image
from pathlib import Path

from audiomentations import Compose, AddGaussianNoise

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


def sound_example_generator(
    sound_file_paths,
    batch_size=8,
    augment=True,
    save_augmented_images_to_path=None,
    fixed_sound_length=FIXED_SOUND_LENGTH,
    num_mels=NUM_MELS,
    preprocessing_fn=None,
):
    if save_augmented_images_to_path:
        os.makedirs(save_augmented_images_to_path, exist_ok=True)

    augmenter = Compose([
        AddGaussianNoise(p=0.1)
    ])

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
                sound_np = augmenter(samples=sound_np, sample_rate=SAMPLE_RATE)

            vectors = preprocess_audio_chunk(
                sound_np, fixed_sound_length=fixed_sound_length, num_mels=num_mels
            )
            if save_augmented_images_to_path:
                # Save the augmented image(vectors) to path
                generated_uuid = uuid.uuid4()
                input_image_pil = Image.fromarray((vectors * 255).astype(np.uint8))
                input_image_pil.save(
                    os.path.join(
                        save_augmented_images_to_path,
                        "{}__{}_input.png".format(
                            Path(sound_file_path).stem, generated_uuid
                        ),
                    )
                )

            x.append(vectors)
            y.append(target)

        x = np.array(x)
        y = np.array(y)

        if preprocessing_fn:
            x = preprocessing_fn(x)

        yield x, y
