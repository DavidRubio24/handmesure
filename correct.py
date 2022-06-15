import os
import sys

import cv2
import numpy as np

from utils.files import open_image

COLOR_SCHEME = [[221, 229, 205], [227, 30, 58], [112, 110, 112], [233, 59, 147], [172, 212, 191], [22, 42, 79], [56, 137, 192], [52, 18, 199], [162, 247, 132], [54, 129, 157], [39, 29, 226], [164, 126, 30], [32, 70, 53], [220, 28, 142], [33, 249, 24], [127, 148, 194], [57, 206, 55], [162, 222, 243], [72, 148, 77], [169, 228, 236], [114, 69, 177], [145, 176, 127], [39, 208, 225], [237, 120, 42], [165, 135, 78], [0, 29, 129], [143, 144, 59], [7, 106, 219], [58, 78, 77], [38, 126, 209], [90, 198, 169], [59, 16, 221], [249, 96, 196], [162, 129, 137], [223, 9, 143], [216, 3, 123], [204, 156, 173], [134, 23, 5], [123, 202, 252], [154, 144, 40], [119, 43, 192], [192, 229, 58], [236, 161, 205], [18, 120, 170], [149, 176, 50], [94, 104, 174], [192, 67, 17], [20, 118, 178], [60, 210, 131], [110, 188, 212]]

COLOR_SCHEME_  = [[171, 233, 180], [235, 213, 199], [166, 231, 229], [166, 210, 175], [212, 217, 171], [202, 180, 240], [199, 211, 189], [217, 245, 225], [163, 242, 246], [208, 189, 172], [252, 228, 231], [213, 167, 211], [207, 208, 211], [247, 237, 187], [163, 251, 182], [237, 166, 187], [160, 243, 185], [181, 225, 179], [207, 160, 242], [164, 222, 190], [172, 164, 249], [196, 198, 224], [246, 226, 220], [208, 209, 171], [249, 195, 157], [206, 209, 234], [161, 193, 183], [237, 228, 189], [179, 245, 227], [178, 163, 194], [239, 214, 156], [253, 162, 212], [195, 231, 242], [232, 190, 244], [249, 179, 201], [184, 164, 252], [184, 193, 167], [243, 221, 183], [239, 165, 218], [222, 226, 239], [204, 184, 179], [199, 201, 229], [229, 241, 200], [238, 172, 244], [237, 248, 188], [223, 254, 195], [217, 175, 187], [176, 174, 210], [163, 209, 235], [204, 230, 242]]


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
        cv2.imshow(self.title, image[self.crop[0]:self.crop[2], self.crop[1]:self.crop[3]])

    def show(self):
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
            return self.show()
        elif k == ord('+'):
            x0, y0, x1, y1 = self.crop
            x0, y0, x1, y1 = (x0 + 10, y0 + 10, x1 - 10, y1 - 10)
            crop = (min(x0, x1 - 11), min(y0, y1 - 11), max(x1, x0 + 11), max(y1, y0 + 11))
            self.crop = tuple(map(round, crop))
            return self.show()
        else:
            return self.show()

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
