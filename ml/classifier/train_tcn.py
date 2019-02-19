import os

from keras.layers import Dense
from keras.models import Input, Model
from keras.optimizers import Adam
from tcn import TCN

from ml.classifier.data_generator import (
    get_train_paths,
    get_validation_paths,
    SoundExampleGenerator)
from ml.settings import DATA_DIR


def get_tcn_model(input_vector_size, target_vector_size=1, num_filters=16, learning_rate=0.002):
    model_input = Input(shape=(None, input_vector_size))
    model_output = TCN(
        return_sequences=False,
        nb_filters=int(num_filters),
        dilations=[1, 2, 4, 8, 16, 32, 64],
        nb_stacks=2,
    )(model_input)
    model_output = Dense(target_vector_size, activation="sigmoid")(model_output)

    model = Model(inputs=[model_input], outputs=[model_output])
    model.compile(
        optimizer=Adam(lr=learning_rate, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    # model.summary()

    return model


if __name__ == "__main__":
    train_paths = get_train_paths()
    train_generator = SoundExampleGenerator(train_paths)
    train_sample_batch_x, train_sample_batch_y = next(train_generator)
    n_timesteps = len(train_sample_batch_x[0])
    input_vector_size = len(train_sample_batch_x[0][0])

    validation_paths = get_validation_paths()
    validation_generator = SoundExampleGenerator(validation_paths, augment=False)

    model = get_tcn_model(input_vector_size)

    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        validation_steps=32,
        steps_per_epoch=128,
        epochs=12,
        shuffle=False,
    )

    os.makedirs(DATA_DIR / "models", exist_ok=True)

    model.save(os.path.join(DATA_DIR / "models", "tcn.h5"))
    print("Model is saved")
