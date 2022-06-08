import sys

import cv2
import numpy as np

from model import detect_hand_model, closed_hand_model, opened_hand_model
from utils.files import get_file, open_image


class Predictor:
    def __init__(self, model='detect_hand', model_factory=detect_hand_model, path='models/'):
        model_path = get_file(model, path)
        self.model = model_factory(model_path)
        self.image_shape = None

    def __call__(self, image: np.ndarray | str):
        image = open_image(image)
        self.image_shape = image.shape
        image = cv2.resize(image, tuple(self.model.input.shape[1:3]))
        image = image.astype(np.float32) / 256
        image = image[np.newaxis, ...]
        prediction = self.model(image)
        if isinstance(prediction, list):
            prediction = [p.numpy() for p in prediction]
        else:
            prediction = prediction.numpy()
        return prediction

    def scale(self, landmarks: np.ndarray, shape: tuple = None, copy=True, current_shape: tuple = None):
        if copy:
            landmarks = landmarks.copy()

        shape = shape or self.image_shape
        if shape is None:
            raise ValueError('No shape provided.')

        current_shape = current_shape or tuple(self.model.input.shape[1:])

        landmarks[..., 0] *= shape[1] / current_shape[0]
        landmarks[..., 1] *= shape[0] / current_shape[1]

        return landmarks

    @staticmethod
    def crop(image: np.ndarray, roi: np.ndarray):
        x_min = np.min(roi[..., 0])
        x_max = np.max(roi[..., 0])
        y_min = np.min(roi[..., 1])
        y_max = np.max(roi[..., 1])

        return image[y_min:y_max, x_min:x_max]


def main(path='./model/'):
    pass


if __name__ == '__main__':
    main(*sys.argv[1:])
