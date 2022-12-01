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
        for file in files[::-1]:
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


def chose_name(name, path='./'):
    """
    Chose a name for a file that doesn't already exist.
    If the name has digits they will be increased.
    """
    new_name = os.path.join(path, name) if path else name
    # Until we generate a new name
    while os.path.isfile(new_name):
        # Increase the digits
        new_name = os.path.join(path, increase_name(name))
    return new_name


def increase_name(name):
    """Increase the digits in a name."""
    # Increase the least significant digit within the name
    for i in range(len(name) - 1, -2, -1):
        if i == -1:
            # There is no digit in the name that can be increased
            name = '1' + name
            break
        if name[i] in '012345678':
            # Increase the digit and we are done
            name = name[:i] + str(int(name[i]) + 1) + name[i + 1:]
            break
        elif name[i] == '9':
            # We still have to increase the next digit
            name = name[:i] + '0' + name[i + 1:]
    return name
