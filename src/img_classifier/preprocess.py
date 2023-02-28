import glob

import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split


def load_data(dir_path, class_name):
    X = []
    y = []
    for img_path in glob.glob(dir_path + "/*.jpg"):
        img = io.imread(img_path)
        if class_name in img_path:
            label = 1
        else:
            label = 0
        X.append(img)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y


def split_data(test_size, dir_path, class_name):
    X = load_data(dir_path, class_name)[0]
    y = load_data(dir_path, class_name)[1]
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
