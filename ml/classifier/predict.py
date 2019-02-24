import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature

from ml.classifier.data_generator import get_validation_paths, SoundExampleGenerator
from ml.classifier.train_mobilenet import (
    num_mels,
    fixed_sound_length,
    preprocess_mobilenet_input,
)
from ml.settings import DATA_DIR
from ml.utils.timer import timer


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


if __name__ == "__main__":
    import os

    random.seed(424)

    model = load_model(os.path.join(DATA_DIR / "models", "mobilenet_v2.h5"))

    validation_paths = get_validation_paths()
    num_validation_sounds = sum(
        len(validation_paths[category]) for category in validation_paths
    )
    print("Found {} validation sounds".format(num_validation_sounds))

    with timer("Generate validation data"):
        validation_generator = SoundExampleGenerator(
            validation_paths,
            batch_size=num_validation_sounds,
            num_mels=num_mels,
            fixed_sound_length=fixed_sound_length,
            preprocessing_fn=preprocess_mobilenet_input,
            augment=False,
        )
        validation_x, validation_y = validation_generator[0]

    with timer("Get predictions"):
        y_predicted = model.predict(validation_x, batch_size=64)
    y_predicted_binarized = np.where(y_predicted >= 0.5, 1, 0)

    # Plot normalized confusion matrix
    cnf_matrix = confusion_matrix(validation_y, y_predicted_binarized)
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix,
        classes=["not laughter", "laughter"],
        normalize=True,
        title="Normalized confusion matrix",
    )
    plt.show()

    # Plot precision-recall curve
    plt.figure()
    average_precision = average_precision_score(validation_y, y_predicted)

    print("Average precision-recall score: {0:0.2f}".format(average_precision))
    precision, recall, _ = precision_recall_curve(validation_y, y_predicted)

    step_kwargs = (
        {"step": "post"} if "step" in signature(plt.fill_between).parameters else {}
    )
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, alpha=0.2, color="b", **step_kwargs)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("2-class Precision-Recall curve: AP={0:0.2f}".format(average_precision))
    plt.show()
