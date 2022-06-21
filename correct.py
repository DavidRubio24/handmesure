import os
import sys

import cv2
import numpy as np

from constants import COLOR_SCHEME, points_interest_closed, points_interest_opened, COLORS
from mesure import mesure_closed, mesure_opened
from utils.files import open_image


class Corrector:
    def __init__(self, image: np.ndarray, points: np.ndarray):
        self.points_original = np.array(points, copy=False)
        self.points = np.array(points, dtype=float, copy=True)
        self.points_colors = np.array(COLOR_SCHEME, np.uint8)
        self.crop = (round(self.points[:, 1].min()) - 100,
                     round(self.points[:, 0].min()) - 300,
                     round(self.points[:, 1].max()) + 100,
                     round(self.points[:, 0].max()) + 300)

        image = open_image(image)
        self.image = image

        self.title = 'Enter o Espacio -> correcto. Borrar -> incorrecto. Esc -> deshacer.'
        self.moving_point = None
        cv2.namedWindow(self.title, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.title, self.on_mouse)

    def show_image(self):
        image = self.image.copy()
        r = 10
        # Draw points
        for (x, y), color in zip(self.points, self.points_colors):
            if y < 0 or x < 0 or y >= image.shape[0] or x >= image.shape[1]:
                continue
            x, y = round(x), round(y)
            image[y - r:y + r + 1, x:x+1] = color
            image[y:y+1, x - r:x + r + 1] = color
            cv2.circle(image, (x, y), r, (255, 255, 255), 2)

        # Closed hand
        if len(self.points) == 15:
            distances = mesure_closed(dict(zip(points_interest_closed, self.points)))
        elif len(self.points) == 23:
            distances = mesure_opened(dict(zip(points_interest_opened, self.points)))
        else:
            distances = {}

        for name, ((x0, y0), (x1, y1)) in distances.items():
            cv2.line(image, (round(x0), round(y0)), (round(x1), round(y1)), COLORS[name], 2)

        cv2.imshow(self.title, image[self.crop[0]:self.crop[2], self.crop[1]:self.crop[3]])

    def show(self):
        while True:
            self.show_image()
            k = cv2.waitKey()
            if k == 27:  # Esc
                self.points = self.points_original.copy()
                return self.show()
            elif k == 8:          # Return
                return False
            elif k in [32, 13]:   # Space or enter
                return np.any(self.points != self.points_original)
            elif k == ord('-'):
                x0, y0, x1, y1 = self.crop
                crop = (max(x0 - 10, 0), max(y0 - 10, 0), min(x1 + 10, self.image.shape[0] - 1), min(y1 + 10, self.image.shape[1] - 1))
                self.crop = tuple(map(round, crop))
            elif k == ord('+'):
                x0, y0, x1, y1 = self.crop
                x0, y0, x1, y1 = (x0 + 10, y0 + 10, x1 - 10, y1 - 10)
                crop = (min(x0, x1 - 11), min(y0, y1 - 11), max(x1, x0 + 11), max(y1, y0 + 11))
                self.crop = tuple(map(round, crop))

    def on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.moving_point = self.closest_point(x + self.crop[1], y + self.crop[0])
            self.points[self.moving_point, :2] = (x + self.crop[1], y + self.crop[0])
            self.show_image()
        elif event == cv2.EVENT_MOUSEMOVE and self.moving_point is not None:
            self.points[self.moving_point, :2] = (x + self.crop[1], y + self.crop[0])
            self.show_image()
        elif event == cv2.EVENT_RBUTTONUP:
            self.moving_point = None
            self.show_image()

    def closest_point(self, x, y):
        distances = np.sqrt((self.points[..., 0] - x) ** 2 + (self.points[..., 1] - y) ** 2)
        return np.argmin(distances)


def main(path='landmarks_M2_00.txt', out='landmarks_M2_01.txt'):
    with open(path) as f:
        landmarks: dict = eval(f.read())
    new_landmarks = landmarks.copy()
    for image_path, points in landmarks.items():
        image = cv2.imread(image_path)
        points = np.array(points)
        c = Corrector(image, points)
        update = c.show()
        if update:
            new_landmarks[image_path] = c.points.tolist()
        cv2.destroyWindow(c.title)
        with open(out, 'w') as f:
            f.write(str(new_landmarks))
    return new_landmarks


if __name__ == '__main__':
    main(*sys.argv[1:])
