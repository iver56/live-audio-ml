import os

import numpy as np
from keras import optimizers
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

from ml.classifier.data_generator import (
    sound_example_generator,
    get_train_paths,
    get_validation_paths,
)
from ml.settings import DATA_DIR


def preprocess_mobilenet_input(x):
    """
    :param x: 3D tensor of spectrograms with values in the range [0, 1]
    :return: 4D tensor of spectrograms with values in the range [-1, 1]. The last channel
     represents RGB.
    """

    # Values should end up in the range [-1, 1]
    scaled_images = (2 * (x - 0.5)).astype(np.float32)

    # Apply the same greyscale image to all three color channels
    rgb_canvases = np.zeros(shape=(x.shape[0], x.shape[1], x.shape[2], 3), dtype=np.float32)
    rgb_canvases[:, :, :, 0] = scaled_images
    rgb_canvases[:, :, :, 1] = scaled_images
    rgb_canvases[:, :, :, 2] = scaled_images
    return rgb_canvases


def get_mobilenet_model(img_width, img_height, target_vector_size=1):
    model = MobileNetV2(
        alpha=0.5, weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)
    )

    # Tune all layers
    for layer in model.layers[:]:
        layer.trainable = True

    # Add custom Layers
    x = model.output
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(target_vector_size, activation="sigmoid")(x)

    model_final = Model(inputs=model.input, outputs=predictions)

    # Compile the model
    model_final.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True),
        metrics=["binary_accuracy"],
    )

    # model.summary()

    return model_final


if __name__ == "__main__":
    num_mels = img_width = 128
    fixed_sound_length = img_height = 128

    train_paths = get_train_paths()
    train_generator = sound_example_generator(
        train_paths,
        num_mels=num_mels,
        fixed_sound_length=fixed_sound_length,
        preprocessing_fn=preprocess_mobilenet_input,
    )

    validation_paths = get_validation_paths()
    validation_generator = sound_example_generator(
        validation_paths,
        num_mels=num_mels,
        fixed_sound_length=fixed_sound_length,
        preprocessing_fn=preprocess_mobilenet_input,
    )

    model = get_mobilenet_model(img_width, img_height)

    os.makedirs(DATA_DIR / "models", exist_ok=True)

    model_save_path = os.path.join(DATA_DIR / "models", "mobilenet_v2.h5")
    model_checkpoint = ModelCheckpoint(
        model_save_path, monitor="val_binary_accuracy", verbose=1, save_best_only=True
    )

    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        validation_steps=192,
        steps_per_epoch=64,
        epochs=20,
        shuffle=False,
        callbacks=[model_checkpoint],
    )
