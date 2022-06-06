import os
import sys

import cv2
import numpy as np


COLOR_SCHEME = [[221, 229, 205], [227, 30, 58], [112, 110, 112], [233, 59, 147], [172, 212, 191], [22, 42, 79], [56, 137, 192], [52, 18, 199], [162, 247, 132], [54, 129, 157], [39, 29, 226], [164, 126, 30], [32, 70, 53], [220, 28, 142], [33, 249, 24], [127, 148, 194], [57, 206, 55], [162, 222, 243], [72, 148, 77], [169, 228, 236], [114, 69, 177], [145, 176, 127], [39, 208, 225], [237, 120, 42], [165, 135, 78], [0, 29, 129], [143, 144, 59], [7, 106, 219], [58, 78, 77], [38, 126, 209], [90, 198, 169], [59, 16, 221], [249, 96, 196], [162, 129, 137], [223, 9, 143], [216, 3, 123], [204, 156, 173], [134, 23, 5], [123, 202, 252], [154, 144, 40], [119, 43, 192], [192, 229, 58], [236, 161, 205], [18, 120, 170], [149, 176, 50], [94, 104, 174], [192, 67, 17], [20, 118, 178], [60, 210, 131], [110, 188, 212]]


class Corrector:
    def __init__(self, image: np.ndarray, points: np.ndarray, window_size=.25, title='', point_idx=0):
        self.points_original = points
        self.points = np.array(points, dtype=float, copy=True)
        self.points_colors = np.array(COLOR_SCHEME, np.uint8)

        self.image = image
        self.resized = image.copy()
        # TODO: zoom in to centain point.
        self.crop = [1500, 0, image.shape[0], image.shape[1]]
        self.crop[1] = round(self.crop[0] * image.shape[1] / image.shape[0])
        self.window_size = (image.shape[0] * window_size, image.shape[1] * window_size)
        self.window_size = (round(self.window_size[0]), round(self.window_size[1]))
        self.update_resized()

        self.title = 'Enter o Espacio -> correcto. Borrar -> incorrecto. Esc -> deshacer.' + title
        self.moving_point = None
        cv2.namedWindow(self.title)
        cv2.setMouseCallback(self.title, self.on_mouse)

    def update_resized(self):
        y0, x0, y1, x1 = map(int, self.crop)
        self.resized = cv2.resize(self.image[y0:y1, x0:x1], self.window_size[::-1], interpolation=cv2.INTER_NEAREST)

    def window2original(self, x, y) -> tuple[float, float]:
        y0, x0, y1, x1 = self.crop
        x = x0 + x * float(x1 - x0) / self.window_size[1]
        y = y0 + y * float(y1 - y0) / self.window_size[0]
        return x, y

    def original2window(self, x, y) -> tuple[float, float]:
        y0, x0, y1, x1 = self.crop
        x = float(x - x0) * self.window_size[1] / (x1 - x0)
        y = float(y - y0) * self.window_size[0] / (y1 - y0)
        return x, y

    def show_image(self):
        resized = self.resized.copy()
        r = 3
        # Draw points
        for (x, y), color in zip(self.points, self.points_colors):
            x, y = self.original2window(x, y)
            if y < 0 or x < 0 or y >= self.window_size[0] or x >= self.window_size[1]:
                continue
            x, y = round(x), round(y)
            resized[y - r:y + r + 1, x:x+1] = color
            resized[y:y+1, x - r:x + r + 1] = color
        cv2.imshow(self.title, resized)

    def show(self):
        self.show_image()
        k = cv2.waitKey(0)
        if k == 27:  # Esc
            self.points = self.points_original.copy()
            return self.show()
        elif k == 8:          # Return
            return False
        elif k in [32, 13]:   # Space or enter
            return True
        elif k == ord('+'):  # Zoom in
            self.window_size = (self.window_size[0] / .9, self.window_size[1] / .9)
            self.window_size = (round(self.window_size[0]), round(self.window_size[1]))
            self.update_resized()
            return self.show()
        elif k == ord('-'):  # Zoom out
            self.window_size = (self.window_size[0] * .9, self.window_size[1] * .9)
            self.window_size = (round(self.window_size[0]), round(self.window_size[1]))
            self.update_resized()
            return self.show()
        else:
            return self.show()

    def on_mouse(self, event, x, y, flags, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.moving_point = self.closest_point(*self.window2original(x, y))
            self.points[self.moving_point, :2] = self.window2original(x, y)
            self.show_image()
        elif event == cv2.EVENT_MOUSEMOVE and self.moving_point is not None:
            self.points[self.moving_point, :2] = self.window2original(x, y)
            self.show_image()
        elif event == cv2.EVENT_LBUTTONUP:
            self.moving_point = None
            self.show_image()
        elif event == cv2.EVENT_MOUSEWHEEL and flags & cv2.EVENT_FLAG_CTRLKEY:
            scale = .95
            if flags < 0:
                x, y = self.window2original(x, y)
                y0, x0, y1, x1 = self.crop
                y0, x0, y1, x1 = y0 - y, x0 - x, y1 - y, x1 - x
                y0, x0, y1, x1 = y0 / scale, x0 / scale, y1 / scale, x1 / scale
                self.crop = y0 + y, x0 + x, y1 + y, x1 + x
                self.crop = (max(0, self.crop[0]),
                             max(0, self.crop[1]),
                             min(self.image.shape[0], self.crop[2]),
                             min(self.image.shape[1], self.crop[3]))
            else:
                x, y = self.window2original(x, y)
                y0, x0, y1, x1 = self.crop
                y0, x0, y1, x1 = y0 - y, x0 - x, y1 - y, x1 - x
                y0, x0, y1, x1 = y0 * scale, x0 * scale, y1 * scale, x1 * scale
                self.crop = y0 + y, x0 + x, y1 + y, x1 + x
                self.crop = (max(0, self.crop[0]),
                             max(0, self.crop[1]),
                             min(self.image.shape[0], self.crop[2]),
                             min(self.image.shape[1], self.crop[3]))
            self.update_resized()
            self.show_image()

    def closest_point(self, x, y):
        distances = np.sqrt((self.points[..., 0] - x)**2 + (self.points[..., 1] - y)**2)
        return np.argmin(distances)

    def __del__(self):
        print('Closing window', self.title)
        cv2.destroyWindow(self.title)


def main(path='landmarks_M2_.txt', out='landmarks_M2_00.txt'):
    with open(path) as f:
        landmarks: dict = eval(f.read())
    new_landmarks = landmarks.copy()
    for image_path, points in landmarks.items():
        image = cv2.imread(image_path)
        points = np.array(points)
        c = Corrector(image, points, title=image_path)
        update = c.show()
        if update:
            new_landmarks[image_path] = c.points.tolist()
        c.__del__()
        with open(out, 'w') as f:
            f.write(str(new_landmarks))
    return new_landmarks
