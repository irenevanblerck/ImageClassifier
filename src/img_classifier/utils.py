# image classification with Tensorflow's Keras (test version)
# load packages
import argparse
import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from scikeras.wrappers import KerasClassifier
from skimage import io
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.optimizers import Adam

# import the default values
DIR_PATH_DEFAULT = ".data/"
CLASS_NAME_DEFAULT = "dusting"
TEST_SIZE_DEFAULT = 0.2
NUM_IMG_DEFAULT = 9
IMG_SIZE_DEFAULT = 256
FLIP_TYPE_DEFAULT = "horizontal"
ROTATION_VALUE_DEFAULT = 0.1
ZOOM_VALUE_DEFAULT = 0.1
BATCH_SIZE_SPACE_DEFAULT = 32
ACTIVATION_SPACE_DEFAULT = "relu"
LEARNING_RATE_SPACE_DEFAULT = 0.01
EPOCHS_SPACE_DEFAULT = 20


# load data
def load_data(DIR_PATH_DEFAULT, CLASS_NAME_DEFAULT):
    # initialize empty lists to store data and labels
    X = []
    y = []

    # iterate over all image files in the directory
    for img_path in glob.glob(DIR_PATH_DEFAULT + "/*.jpg"):
        # read the image in RGB format
        img = io.imread(img_path)

        # determine the label of the image based on whether the class name appears in the filename
        if CLASS_NAME_DEFAULT in img_path:
            label = 1
        else:
            label = 0

        # append the image data and label to their respective lists
        X.append(img)
        y.append(label)

    # convert the data and labels to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # return the data and labels as a tuple
    return X, y


# split data
def split_data(TEST_SIZE_DEFAULT):
    # Load the data from a directory path and class name
    X = load_data(DIR_PATH_DEFAULT, CLASS_NAME_DEFAULT)[0]
    y = load_data(DIR_PATH_DEFAULT, CLASS_NAME_DEFAULT)[1]

    # Set the random state for reproducibility
    random_state = 51

    # Split the data into training and testing sets using train_test_split
    # with a specified test size and random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE_DEFAULT, random_state=random_state
    )

    # Return the training and testing data
    return X_train, X_test, y_train, y_test


# This function displays a specified number of images
def display_img(NUM_IMG_DEFAULT):
    # Create a new figure with specified size
    fig = plt.figure(figsize=(10, 7))

    # Get a random sample of indices from the test data
    lst = random.sample(
        range(0, len(split_data(TEST_SIZE_DEFAULT)[1])), NUM_IMG_DEFAULT
    )

    # Loop through each index and display the corresponding image
    for index, value in enumerate(lst):
        # Add a new subplot to the figure and set its position
        fig.add_subplot(330 + 1 + index)

        # Display the image corresponding to the current index
        plt.imshow(split_data(TEST_SIZE_DEFAULT)[0][value])

    # Show the figure with all the subplots
    plt.show()


def rescale_resize(IMG_SIZE_DEFAULT):
    aug = tensorflow.keras.Sequential(
        [
            tensorflow.keras.layers.Resizing(IMG_SIZE_DEFAULT, IMG_SIZE_DEFAULT),
            tensorflow.keras.layers.Rescaling(1.0 / 255),
        ]
    )
    return aug


def random_flip(FLIP_TYPE_DEFAULT):
    aug = tensorflow.keras.layers.RandomFlip(FLIP_TYPE_DEFAULT)
    return aug


def random_rotation(ROTATION_VALUE_DEFAULT):
    aug = tensorflow.keras.Sequential(
        [tensorflow.keras.layers.RandomRotation(ROTATION_VALUE_DEFAULT)]
    )
    return aug


def random_zoom(ZOOM_VALUE_DEFAULT):
    aug = tensorflow.keras.Sequential(
        [tensorflow.keras.layers.RandomZoom(ZOOM_VALUE_DEFAULT)]
    )
    return aug


# create model
def create_model(learning_rate, activation):
    # set the logging level to ERROR to avoid warnings
    tensorflow.get_logger().setLevel("ERROR")

    # create a sequential model with Keras
    model = tensorflow.keras.models.Sequential(
        [
            Input(shape=(None, 256, 3)),
            rescale_resize(IMG_SIZE_DEFAULT),
            random_flip(FLIP_TYPE_DEFAULT),
            random_zoom(ZOOM_VALUE_DEFAULT),
            random_rotation(ROTATION_VALUE_DEFAULT),
            Dense(32, input_shape=(65536,), activation=activation),
            Dense(64, activation=activation),
            Dense(128, activation=activation),
            Dense(64, activation=activation),
            Dense(1, activation="sigmoid"),
        ]
    )

    # compile the model with binary crossentropy loss and accuracy metric
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # save the model
    model.save("./model.h5")

    # load and return the model
    model = load_model("./model.h5")

    return model


# Create a KerasClassifier
def hyperparameter_tuning():
    # Create a KerasClassifier
    model = KerasClassifier(
        build_fn=create_model,
        learning_rate=[0.1, 0.01],
        activation=["relu", "tanh"],
        verbose=1,
    )

    # Define the search space
    params = {
        "activation": ["relu", "tanh"],
        "batch_size": [32, 128, 256],
        "epochs": [10],
        "learning_rate": [0.1, 0.01],
    }

    # Create a randomize search cv object passing in the parameters to try
    random_search = RandomizedSearchCV(model, params, cv=3)

    # Fit the model
    random_search.fit(
        split_data(TEST_SIZE_DEFAULT)[0], split_data(TEST_SIZE_DEFAULT)[2]
    )

    # Print the tuned parameters and score
    print("Tuned CNN Parameters: {}".format(random_search.best_params_))
    print("Best score is {}".format(random_search.best_score_))

    return random_search


# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Image classification with Keras.")
    parser.add_argument(
        "--dir_path",
        type=str,
        required=False,
        default=DIR_PATH_DEFAULT,
        help="Directory path to the images",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        required=False,
        default=CLASS_NAME_DEFAULT,
        help="Class name of the images",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        required=False,
        default=TEST_SIZE_DEFAULT,
        help="Size of the test set",
    )
    parser.add_argument(
        "--num_img",
        type=int,
        required=False,
        default=NUM_IMG_DEFAULT,
        help="Number of images to display",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        required=False,
        default=IMG_SIZE_DEFAULT,
        help="Size of the images",
    )
    parser.add_argument(
        "--flip_type",
        type=str,
        required=False,
        default=FLIP_TYPE_DEFAULT,
        help="Type of flip",
    )
    parser.add_argument(
        "--rotation_value",
        type=float,
        required=False,
        default=ROTATION_VALUE_DEFAULT,
        help="Value of rotation",
    )
    parser.add_argument(
        "--zoom_value",
        type=float,
        required=False,
        default=ZOOM_VALUE_DEFAULT,
        help="Value of zoom",
    )
    args = parser.parse_args()
    return args


# main function
def main():
    args = argument_parser()
    load_data(args.dir_path, args.class_name)
    split_data(args.test_size)
    display_img(args.num_img)
    rescale_resize(args.img_size)
    random_flip(args.flip_type)
    random_rotation(args.rotation_value)
    random_zoom(args.zoom_value)
    create_model(LEARNING_RATE_SPACE_DEFAULT, ACTIVATION_SPACE_DEFAULT)
    hyperparameter_tuning()


if __name__ == "__main__":
    main()
