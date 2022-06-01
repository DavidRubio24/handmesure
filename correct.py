import os
import sys

import cv2
import numpy as np


class Corrector:
    def __init__(self, image: np.ndarray, points: np.ndarray, scale=.3):
        self.image = image
        self.resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        self.points_original = points
        self.points = np.array(points, dtype=int, copy=True)
        self.points_colors = np.array([[255, 255, 255] for point in points], dtype=np.uint8)
        self.scale = scale
        self.title = 'Enter -> correcto. Espacio -> incorrecto. Esc -> deshacer.'
        self.moving = False
        self.moving_point = None
        cv2.namedWindow(self.title)
        cv2.setMouseCallback(self.title, self.on_mouse)

    def ready_image(self):
        self.resized = cv2.resize(self.image, (0, 0), fx=self.scale, fy=self.scale)

        # Draw points
        for (x, y), color in zip(self.points, self.points_colors):
            x, y = round(x * self.scale), round(y * self.scale)
            cv2.circle(self.resized, (x, y), 5, tuple(map(int, color)), -1)
        return self.resized

    def show(self):
        self.ready_image()
        cv2.imshow(self.title, self.resized)
        k = cv2.waitKey(0)
        if k == 27:  # Esc
            self.points = self.points_original.copy()
            return self.show()
        elif k == 32:        # Space
            return False
        elif k == 13:        # Enter
            return True
        elif k == ord('+'):  # Zoom in
            self.scale += .1
            return self.show()
        elif k == ord('-'):  # Zoom out
            self.scale -= .1
            return self.show()

    def on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.moving = True
            self.moving_point = self.closest_point(x / self.scale, y / self.scale)
            # set color to red
            self.points_colors[self.moving_point] = [0, 0, 255]
            self.ready_image()
            cv2.imshow(self.title, self.resized)
        elif event == cv2.EVENT_MOUSEMOVE and self.moving and self.moving_point is not None:
            self.points[self.moving_point, 0] = x / self.scale
            self.points[self.moving_point, 1] = y / self.scale
            self.ready_image()
            cv2.imshow(self.title, self.resized)
        elif event == cv2.EVENT_LBUTTONUP:
            self.moving = False
            self.moving_point = None
            # set color to white
            self.points_colors[:] = [255, 255, 255]
            self.ready_image()
            cv2.imshow(self.title, self.resized)

    def closest_point(self, x, y):
        distances = np.sqrt((self.points[..., 0] - x)**2 + (self.points[..., 1] - y)**2)
        return np.argmin(distances)


def correct(image: np.ndarray, points: np.ndarray, scale=.3,
            title: str = 'Enter -> correcto. Espacio -> incorrecto. Esc -> deshacer.'):
    cv2.namedWindow(title)
    # cv2.setMouseCallback(title, on_mouse, image)
    reduced = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow(title, reduced)
    k = cv2.waitKey(0)
    cv2.destroyWindow(title)
    if k == 27:  # Esc
        return correct(image, points)
    return k == 13, points


def main(images_folder='./data/', pointss: np.ndarray | str = './data/points.npy'):
    if isinstance(pointss, str):
        pointss = np.load(pointss)

    images = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith(('.png', '.bmp', '.jpg'))]
    if len(images) != len(pointss):
        raise ValueError('Number of images and points do not match.')

    for image_path, points in zip(images, pointss):
        img = cv2.imread(image_path)
        good, corrected = correct(img, points)
        points[:] = corrected


if __name__ == '__main__':
    main(*sys.argv[1:])
