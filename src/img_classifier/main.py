import argparse

from img_classifier.config import (
    DIR_PATH_DEFAULT,
    CLASS_NAME_DEFAULT,
    TEST_SIZE_DEFAULT,
    NUM_IMG_DEFAULT,
    IMG_SIZE_DEFAULT,
    FLIP_TYPE_DEFAULT,
    ROTATION_VALUE_DEFAULT,
    ZOOM_VALUE_DEFAULT,
    ACTIVATION_SPACE_DEFAULT,
    LEARNING_RATE_SPACE_DEFAULT,
)
from img_classifier.preprocess import load_data, split_data
from img_classifier.train import create_model, hyperparameter_tuning
from img_classifier.visualize import display_img
from img_classifier.augmentation import (
    rescale_resize,
    random_flip,
    random_rotation,
    random_zoom,
)


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Image classification with Keras.")
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
