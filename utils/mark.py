import os
import sys

import cv2

from images import bar
from predict import Predictor
from model import closed_hand_model, opened_hand_model


def original_points(landmarks='landmarks_M2.txt', out='landmarks_M2_.txt', data='data.txt'):
    with open(landmarks, 'r') as f:
        l1 = eval(f.read())
    with open(data, 'r') as f:
        d = eval(f.read())

    l1_ = {}
    for image, points in l1.items():
        original_image = './data/' + image[12:-9] + '.png'
        (x0, y0), (x1, y1) = d[original_image]
        x0 = min(x0, x1)
        y0 = min(y0, y1)
        l1_[original_image] = [(x + x0, y + y0) for x, y in points]

    with open(out, 'w') as f:
        f.write(str(l1_))

    return l1_


def main(path='./data/crop'):
    images = [f'{path}/{file}' for file in os.listdir(path) if file.endswith('.png') and 'M2' in file]
    d = {}
    predictor = Predictor('open', model_factory=opened_hand_model)
    for image in bar(images, append='{0}'):
        i = cv2.imread(image)
        landmarks = predictor(i)
        landmarks = predictor.scale(landmarks.reshape(-1, 2), i.shape[:2][::-1])
        d[image] = [list(x) for x in landmarks]
        # Save d to file.
        with open('landmarks.txt', 'w') as f:
            f.write(str(d))

    return d


if __name__ == '__main__':
    main(*sys.argv[1:])
