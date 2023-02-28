import random

import matplotlib.pyplot as plt

from img_classifier.preprocess import split_data


def display_img(num_img, test_size):
    fig = plt.figure(figsize=(10, 7))
    lst = random.sample(range(0, len(split_data(test_size)[1])), num_img)
    for index, value in enumerate(lst):
        fig.add_subplot(330 + 1 + index)
        plt.imshow(split_data(test_size)[0][value])
    plt.show()
