import os

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, TimeDistributed, Dropout, Activation, CuDNNLSTM
from keras.optimizers import RMSprop

from ml.classifier.data_generator import (
    sound_example_generator,
    get_train_paths,
    get_validation_paths,
)
from ml.settings import DATA_DIR


def get_lstm_model(input_vector_size, target_vector_size=1, dropout0=0.1, learning_rate=0.001):
    model = Sequential()
    model.add(
        TimeDistributed(Dense(32, activation="relu"), input_shape=(None, input_vector_size))
    )
    model.add(TimeDistributed(Dropout(dropout0)))
    model.add(CuDNNLSTM(units=32, return_sequences=True))
    model.add(Activation("relu"))
    model.add(CuDNNLSTM(units=32, return_sequences=False))
    model.add(Activation("relu"))
    model.add(Dense(target_vector_size, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(lr=learning_rate),
        metrics=["binary_accuracy"],
    )
    # model.summary()
    return model


if __name__ == "__main__":
    train_paths = get_train_paths()
    train_generator = sound_example_generator(train_paths, fixed_sound_length=300, num_mels=20)
    train_sample_batch_x, train_sample_batch_y = next(train_generator)
    n_timesteps = len(train_sample_batch_x[0])
    input_vector_size = len(train_sample_batch_x[0][0])

    validation_paths = get_validation_paths()
    validation_generator = sound_example_generator(validation_paths, fixed_sound_length=300, num_mels=20)

    model = get_lstm_model(input_vector_size)

    os.makedirs(DATA_DIR / "models", exist_ok=True)

    model_save_path = os.path.join(DATA_DIR / "models", "lstm.h5")
    model_checkpoint = ModelCheckpoint(
        model_save_path, monitor="val_binary_accuracy", verbose=1, save_best_only=True
    )

    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        validation_steps=32,
        steps_per_epoch=192,
        epochs=50,
        shuffle=False,
        callbacks=[model_checkpoint],
    )
