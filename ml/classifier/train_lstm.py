import os

import joblib
from keras import Sequential
from keras.layers import Dense, TimeDistributed, Dropout, Activation, LSTM, CuDNNLSTM
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
    data = joblib.load(os.path.join(DATA_DIR / "prepared_dataset", "dataset.pkl"))
    x_sequences = data["x_sequences"]
    y_values = data["y_values"]

    n_timesteps = len(x_sequences[0])
    input_vector_size = len(x_sequences[0][0])

    model = get_lstm_model(input_vector_size)

    train_paths = get_train_paths()
    train_generator = sound_example_generator(train_paths)
    validation_paths = get_validation_paths()
    validation_generator = sound_example_generator(validation_paths)

    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        validation_steps=64,
        steps_per_epoch=128,
        epochs=12,
        shuffle=False,
    )

    os.makedirs(DATA_DIR / "models", exist_ok=True)

    model.save(os.path.join(DATA_DIR / "models", "tcn.h5"))
    print("Model is saved")
