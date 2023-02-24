import os

import cv2
import numpy as np

from constants import points_interest_opened, points_interest_closed


class bar:
    def __init__(self, *iterable, append='', length=None, bar_size=100):
        """Takes as positional arguments the same as range or just an iterable object."""
        if not iterable: raise TypeError('bar expected at least 1 argument, got 0')
        iterable = range(*iterable) if isinstance(iterable[0], int) else iterable[0]
        self.iterable, self.append, self.start, self.index, self.item, self.bar_size, self.len = iter(iterable), append, False, 0, None, bar_size, len(iterable) if length is None else length

    def __iter__(self): return self

    def __next__(self):
        from time import monotonic
        now = monotonic(); self.start = self.start or now; took = now - self.start
        print('\r{:_}/{:<} ({:3}%) '.replace('_', str(len(str(self.len)))).format(self.index, self.len, int(100 * self.index / self.len)), end='')
        if self.index >= self.len:
            print(f"\x1B[0;34m[\x1B[0;32m{'■' * self.bar_size}\x1B[0;34m]\x1B[0m  Took:{int(took / 3600): 3}:{int(took / 60) % 60:02}:{took % 60:04.1f} " + self.append.format(item=self.item, index=self.index-1))
        self.item = next(self.iterable)
        eta = ((self.len - self.index) * took / self.index) if self.index else -1
        done = self.bar_size * self.index / self.len
        print("[\x1B[0;32m{:·<_}]   ETA:".replace('_', str(self.bar_size + 4)).format('■' * int(done) + str(int(10 * (done % 1))) + '\x1B[0m')
              + (f'{int(eta / 3600): 3}:{int(eta / 60) % 60:02}:{eta % 60:04.1f} ' if self.index else ' ' * 12) + self.append.format(item=self.item, index=self.index), end='')
        self.index += 1
        return self.item


def image_cropper(directory=r'\\10.10.204.24\scan4d\TENDER\HANDS_CALIBRADAS\REVISADAS/', output='data/train/',
                  formats='.png', start=0):
    """Takes the images from a directory and crops them around the hand."""
    files = [file for file in os.listdir(directory) if file.endswith(formats) and not file.endswith('mesures.png')]
    from mediapipe.python.solutions.hands import Hands
    hands = Hands().__enter__()
    for file in bar(files[start:], append='{item}'):
        crop = [(0, 0), (-1, -1)]
        image = cv2.imread(os.path.join(directory, file))
        results = hands.process(image[..., ::-1])
        if results is not None and results.multi_hand_landmarks is not None:
            landmarks = results.multi_hand_landmarks[0].landmark
            if landmarks is not None:
                lms = np.array([(l.x, l.y) for l in landmarks])
                lms[:, 0] *= image.shape[1]
                lms[:, 1] *= image.shape[0]

                crop = [(max(round(lms[:, 0].min()) - 200, 0), max(round(lms[:, 1].min()) - 100, 0)),
                        (min(round(lms[:, 0].max()) + 100, image.shape[1]), min(round(lms[:, 1].max()) + 100, image.shape[1]))]

        # Crop the image around the hand.
        image = image[crop[0][1]:crop[1][1],
                      crop[0][0]:crop[1][0]]
        cv2.imwrite(os.path.join(output, file[:-3] +  str(crop) + '.png'), image)


def get_crop_info(directory='data/train'):
    return {f[:f.find('.')]: eval(f[f.find('['):f.find(']') + 1]) for f in os.listdir(directory) if f.endswith('.png')}


def get_labels(directory_in=r'\\10.10.204.24\scan4d\TENDER\HANDS_CALIBRADAS\REVISADAS/', directory_out='data/train'):
    """Takes the JSONs from a directory and saves them as npy after "cropping" the points."""
    crops = get_crop_info(directory_out)
    files = [file for file in os.listdir(directory_in) if file.endswith('.json')]
    # Find corresponding file.
    for file, crop in bar(crops.items()):
        for f in files[::-1]:
            if file in f:
                break
        else:
            print(f'No se ha encontrado el archivo correspondiente a {file} en {directory_in}.')
            continue
        # Read the file.
        d = eval(open(os.path.join(directory_in, f)).read())
        # Get only the points. Take the absolute value, that way the deleted points are also used (better than nothing).
        landmarks = np.abs([v for k, v in d.items() if k in points_interest_opened or k in points_interest_closed])
        landmarks[..., 0] -= crop[0][0]
        landmarks[..., 1] -= crop[0][1]
        np.save(os.path.join(directory_out, file + '.npy'), np.array(landmarks, dtype=int))


def resize(directory='data/train', size=(224, 224)):
    """Resizes the images in a directory to a given size. It also resizes the labels."""
    keyword = f'{size[0]}x{size[1]}'
    for file in bar(os.listdir(directory)):
        if file.endswith('.png') and keyword not in file:
            img = cv2.imread(os.path.join(directory, file))
            image = cv2.resize(img, size)
            cv2.imwrite(os.path.join(directory, file[:-3] + f'{keyword}.png'), image)
            labels = np.load(os.path.join(directory, file[:file.find('.')] + '.npy')).astype(float)
            labels[..., 0] *= size[0] / img.shape[1]
            labels[..., 1] *= size[1] / img.shape[0]
            np.save(os.path.join(directory, file[:file.find('.')] + f'.{keyword}.npy'), labels)


def unite_labels(directory='data/train', type='M1', size=(224, 224)):
    """Unites the labels in a single file."""
    labels = []
    mapping = {}
    index = 0
    keyword = f'{size[0]}x{size[1]}'
    for file in bar(os.listdir(directory)):
        if file.endswith('.npy') and type in file and keyword in file and not 'labels' in file:
            labels.append(np.load(os.path.join(directory, file)))
            mapping[file[:file.find('.')]] = index
            index += 1
    np.save(os.path.join(directory, f'labels.{type}.npy'), np.array(labels))
    with open(os.path.join(directory, f'mapping.{type}.json'), 'w') as f:
        f.write(str(mapping).replace('\'', '"').replace(', ', ',\n'))
