from scikeras.wrappers import KerasClassifier
from logging import Logger
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from img_classifier.preprocess import split_data
from img_classifier.augmentation import (
    rescale_resize,
    random_flip,
    random_rotation,
    random_zoom,
)


def create_model(
    learning_rate, activation, img_size, flip_type, zoom_value, rotation_value
):
    Logger.get_logger().setLevel("ERROR")
    model = Sequential(
        [
            Input(shape=(None, 256, 3)),
            rescale_resize(img_size),
            random_flip(flip_type),
            random_zoom(zoom_value),
            random_rotation(rotation_value),
            Dense(32, input_shape=(65536,), activation=activation),
            Dense(64, activation=activation),
            Dense(128, activation=activation),
            Dense(64, activation=activation),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.save("./model.h5")
    model = load_model("./model.h5")
    return model


def hyperparameter_tuning(test_size):
    # Create a KerasClassifier
    model = KerasClassifier(
        build_fn=create_model,
        learning_rate=[0.1, 0.01],
        activation=["relu", "tanh"],
        verbose=1,
    )
    params = {
        "activation": ["relu", "tanh"],
        "batch_size": [32, 128, 256],
        "epochs": [10],
        "learning_rate": [0.1, 0.01],
    }
    random_search = RandomizedSearchCV(model, params, cv=3)
    random_search.fit(split_data(test_size)[0], split_data(test_size)[2])
    print("Tuned CNN Parameters: {}".format(random_search.best_params_))
    print("Best score is {}".format(random_search.best_score_))
    return random_search
