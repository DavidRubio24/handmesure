import os
import sys

import cv2
import numpy as np


class Cropper:
    def __init__(self, image, resize_factor=.25):
        self.image = image
        self.points = []
        self.resize_factor = resize_factor

        self.resized = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        cv2.imshow('Cropper', self.resized)
        cv2.setMouseCallback('Cropper', self.on_mouse)
        cv2.waitKey()

    def show_image(self):
        resized = self.resized.copy()
        if len(self.points) == 1:
            x, y = self.points[0]
            cv2.circle(resized, (round(x * self.resize_factor), round(y * self.resize_factor)), 3, (0, 0, 255), -1)
        elif len(self.points) == 2:
            x0, y0 = self.points[0]
            x1, y1 = self.points[1]
            cv2.rectangle(resized, (round(x0 * self.resize_factor), round(y0 * self.resize_factor)),
                                   (round(x1 * self.resize_factor), round(y1 * self.resize_factor)), (0, 0, 255), 1)
        else:
            for x, y in self.points:
                cv2.circle(resized, (round(x * self.resize_factor), round(y * self.resize_factor)), 3, (0, 0, 255), -1)
        cv2.imshow('Cropper', resized)

    def on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((round(x / self.resize_factor), round(y / self.resize_factor)))
            self.points = self.points[-2:]
            self.show_image()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.pop()
            self.show_image()


def main(path='./data'):
    images = [f'{path}/{file}' for file in os.listdir(path) if file.endswith('.png')]
    d = {}
    for image in images:
        c = Cropper(cv2.imread(image))
        d[image] = c.points
        # Save d to file.
        with open('data.txt', 'w') as f:
            f.write(str(d))

    return d


def crop():
    data = 'data.txt'
    with open(data, 'r') as f:
        d = eval(f.read())
        for image_path, points in d.items():
            print('Cropping', image_path)
            (x0, y0), (x1, y1) = points
            x0, y0, x1, y1 = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
            image = cv2.imread(image_path)
            image = image[y0:y1, x0:x1]
            cv2.imwrite(image_path[:-4] + '.crop.png', image)


if __name__ == '__main__':
    main(*sys.argv[1:])
