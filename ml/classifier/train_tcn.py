import os
import joblib

from ml.classifier.data_generator import sound_example_generator
from ml.settings import DATA_DIR
from keras.layers import Dense
from keras.models import Input, Model
from keras.optimizers import Adam

from tcn import TCN


def get_tcn_model(input_vector_size, target_vector_size=1, num_filters=16, learning_rate=0.002):
    model_input = Input(shape=(None, input_vector_size))
    model_output = TCN(
        return_sequences=False,
        nb_filters=int(num_filters),
        dilations=[1, 2, 4, 8, 16, 32, 64],
        nb_stacks=2
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
    data = joblib.load(os.path.join(DATA_DIR / "prepared_dataset", "dataset.pkl"))
    x_sequences = data["x_sequences"]
    y_values = data["y_values"]

    n_timesteps = len(x_sequences[0])
    input_vector_size = len(x_sequences[0][0])

    model = get_tcn_model(input_vector_size)

    generator = sound_example_generator()

    model.fit_generator(generator, steps_per_epoch=128, epochs=12, shuffle=False)

    os.makedirs(DATA_DIR / "models", exist_ok=True)

    model.save(os.path.join(DATA_DIR / "models", "tcn.h5"))
    print("Model is saved")
