import os
import cv2
import yaml
import numpy as np


class bar:
    def __init__(self, *iterable, append=''):
        """Inputs the same arguments as range or an iterable object."""
        if not iterable: raise ValueError()
        iterable = range(*iterable) if isinstance(iterable[0], int) else iterable[0]
        self.iterable, self.append, self.start, self.len, self.i, self.r = iter(iterable), append, False, len(iterable), 0, None

    def __iter__(self): return self

    def __next__(self):
        from time import time
        now = time()
        self.start = self.start if self.start else now
        took = int(now - self.start)
        if self.i >= self.len:
            print('\r' + f"{self.i: >6}/{self.len:<} \x1B[0;34m[\x1B[0;32m{'■' * 30}\x1B[0;34m]\x1B[0m  Took:" + ( f'{took // 60: 3}m' if took >= 60 else '    ') + f'{took % 60:3}s  ' + self.append.format(self.r, self.i))
        self.r = next(self.iterable)
        eta = int((self.len - self.i) * (now - self.start) / self.i) if self.i else -1
        done = 30 * self.i / self.len; cs = {0: '', 1: '·', 2: '◧', 3: '■'}  # □ (colab), ◧ (any other console)
        print('\r' + "{: 6}/{:<} [\x1B[0;32m{:·<34}]  ".format(self.i, self.len, '■' * int(done) + cs[int(3 * (done % 1))] + '\x1B[0m')
              + ('ETA:' + (f'{eta // 60: 4}m' if eta >= 60 else '     ') + f'{eta % 60:3}s  ' if self.i else '  ')
              + self.append.format(self.r, self.i), end='')
        self.i += 1
        return self.r


def true(*x): return True


def map_folderwise(function, path, shape=(), dtype=np.float32, condition=true, info=None):
    files = [path + '/' + file for file in os.listdir(path) if condition(path + '/' + file)]

    result = np.zeros((len(files), *shape), dtype)

    for i, file in enumerate(bar(files)):
        if condition(file):
            if info is None:
                result[i] = function(file)
            else:
                result[i] = function(file, info[i])
    return result


def map_arraywise(function, array, shape=(), dtype=None, condition=true):
    dtype = array.dtype if dtype is None else dtype
    result = np.zeros((len(array), *shape), dtype)

    for i, element in enumerate(bar(array)):
        if condition(element):
            result[i] = function(element)
    return result


def add_margin(element):
    x0, x1, y0, y1 = element
    x0 = max(0,    x0 - 100)
    x1 = min(2550, x1 + 100)
    y0 = max(0,    y0 - 100)
    y1 = min(3510, y1 + 100)
    return x0, x1, y0, y1


def crop(file, points, shape=(448, 448)):
    x0, x1, y0, y1 = points
    img = cv2.imread(file)[y0:y1, x0:x1]
    return cv2.resize(img.astype(np.float64) / 256, shape).astype(np.float16)


def labels(file):
    with open(file) as file:
        data = yaml.safe_load(file.read(1446))
    return np.array(list(data.values()))


def imshow(image, destroy=True, name='Image', delay=None):
    cv2.imshow(name, image)
    cv2.waitKey(delay)
    if destroy:
        cv2.destroyWindow(name)


def drawContour(image, contour, colors):
    for (x, y), color in zip(contour, colors):
        image[y, x] = color
