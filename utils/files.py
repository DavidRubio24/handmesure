import os

import cv2
import numpy as np


def identity(x): return x


def get_file(file, path=None, function=identity):
    # Check if file contains path
    if path is not None:
        file = os.path.join(path, file)

    head, tail = os.path.split(file)
    if head == '':
        head = '.'

    definitive_file = None
    files = os.listdir(head)
    if tail in files:
        definitive_file = os.path.join(head, tail)
    else:
        for file in files:
            if file.startswith(tail):
                definitive_file = os.path.join(head, file)
                break

    return function(definitive_file)


def open_image(file, path=None) -> np.ndarray:
    if isinstance(file, str):
        return get_file(file, path, cv2.imread)
    elif isinstance(file, np.ndarray):
        return file
    elif isinstance(file, cv2.VideoCapture):
        return file.read()[1]
    else:
        raise ValueError('Unknown file type, neither a path nor a numpy array.')