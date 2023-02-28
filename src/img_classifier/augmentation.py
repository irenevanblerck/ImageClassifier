from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def rescale_resize(img_size):
    aug = Sequential(
        [
            layers.Resizing(img_size, img_size),
            layers.Rescaling(1.0 / 255),
        ]
    )
    return aug


def random_flip(flip_type):
    aug = layers.RandomFlip(flip_type)
    return aug


def random_rotation(rotation_value):
    aug = Sequential(
        [layers.RandomRotation(rotation_value)]
    )
    return aug


def random_zoom(zoom_value):
    aug = Sequential(
        [layers.RandomZoom(zoom_value)]
    )
    return aug
