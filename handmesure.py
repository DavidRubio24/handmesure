import os.path
import sys

import cv2
import numpy as np

from correct import Corrector
from utils.files import chose_name


def get_landmarks(image: str):
    from predict import Predictor
    from model import closed_hand_model, opened_hand_model
    if 'M1' in image:
        predictor = Predictor(model='closed', model_factory=closed_hand_model)
    elif 'M2' in image:
        predictor = Predictor(model='opened', model_factory=opened_hand_model)
    else:
        print(f'{image} no contiene ni "M1" ni "M2" en el nombre. No se pueden generar los landmarks.')
        return None

    image = cv2.imread(image)

    from mediapipe.python.solutions.hands import Hands
    with Hands() as hands:
        results = hands.process(image[..., ::-1])
    if results is None:
        crop = [(0, 0), (-1, -1)]
    else:
        lms = np.array([(l.x, l.y) for l in results.multi_hand_landmarks[0].landmark])
        lms[:, 0] *= image.shape[1]
        lms[:, 1] *= image.shape[0]

        crop = [(round(lms[:, 0].min()) - 100, round(lms[:, 1].min()) - 100),
                (round(lms[:, 0].max()) + 100, round(lms[:, 1].max()) + 100)]

    image = image[crop[0][1]:crop[1][1],
                  crop[0][0]:crop[1][0]]

    landmarks = predictor(image)

    landmarks = landmarks.reshape(-1, 2)
    landmarks = predictor.scale(landmarks)
    landmarks[..., 0] += crop[0][0]
    landmarks[..., 1] += crop[0][1]

    return landmarks


def main(path='./data'):
    if not os.path.isdir(path):
        print('Ruta inexistente.')
        return

    images = [f'{file}' for file in os.listdir(path) if file.endswith('.png')]

    for image in images:
        landmarks_file_start = image[:-4] + '_lm'
        landmarks_files = sorted([file for file in os.listdir(path) if file.startswith(landmarks_file_start)])
        if landmarks_files:
            landmarks = np.loadtxt(os.path.join(path, landmarks_files[-1]))
            new_landmarks_file = chose_name(landmarks_files[-1])
        else:
            print(f'{image} no tiene landmarks. Se generarán automaticamente.')
            landmarks = get_landmarks(os.path.join(path, image))
            if landmarks is None:
                continue
            new_landmarks_file = chose_name(f'{landmarks_file_start}00.txt')
        c = Corrector(cv2.imread(os.path.join(path, image)), landmarks)
        print('Update landmarks...')
        update = c.show()
        c.__del__()
        if update:
            np.savetxt(os.path.join(path, new_landmarks_file), c.points, fmt='%g')


if __name__ == '__main__':
    main(*sys.argv[1:2])
