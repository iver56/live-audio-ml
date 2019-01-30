import os

import joblib
from keras.models import load_model

from ml.settings import DATA_DIR

if __name__ == "__main__":
    # enforce CPU mode for speedier startup time
    import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = load_model(os.path.join(DATA_DIR / 'models', 'tcn.h5'))

    data = joblib.load(os.path.join(DATA_DIR / "prepared_dataset", 'dataset.pkl'))
    x_sequences = data['x_sequences']
    y_values = data['y_values']

    y_predicted = model.predict(x_sequences)
    for i, y in enumerate(y_predicted):
        print('{:.1f}, {:.1f}'.format(float(y), y_values[i]))
