import os.path
import sys

import cv2
import numpy as np

from constants import points_interest_closed, points_interest_opened
from correct import Corrector
from mesure import mesure_closed, mesure_opened, to_list, compute_distances
from utils.files import chose_name


def get_landmarks(image: str):
    from predict import Predictor
    from model import closed_hand_model, opened_hand_model
    if 'M1' in image:
        predictor = Predictor(model='closed', model_factory=closed_hand_model)
    elif 'M2' in image:
        predictor = Predictor(model='opened', model_factory=opened_hand_model)
    else:
        print(f'{image} no contiene ni "M1" ni "M2" en el nombre. No se pueden generar los landmarks.', file=sys.stderr)
        return None

    image = cv2.imread(image)

    # Use MediaPipe Hands to detect where the hand is.
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

    # Crop the image around the hand.
    image = image[crop[0][1]:crop[1][1],
                  crop[0][0]:crop[1][0]]

    # Detect our landmarks.
    landmarks = predictor(image)

    landmarks = landmarks.reshape(-1, 2)
    landmarks = predictor.scale(landmarks)
    landmarks[..., 0] += crop[0][0]
    landmarks[..., 1] += crop[0][1]

    return landmarks


def main(path=r'.\data'):
    if not os.path.isdir(path):
        print('Ruta inexistente.', file=sys.stderr)
        return

    images = [file for file in os.listdir(path) if file.endswith('.png')]

    for image in images:
        landmarks_file_start = image[:-4] + '_lm'
        landmarks_files = sorted([file for file in os.listdir(path) if file.startswith(landmarks_file_start)])
        if landmarks_files:
            landmarks_file = os.path.join(path, landmarks_files[-1])
            d = eval(open(landmarks_file).read())
            landmarks = np.array([v for k, v in d.items() if k in points_interest_opened or k in points_interest_closed])
            print(f'Leyendo landmarks de {landmarks_file}.')
            new_landmarks_file = chose_name(landmarks_files[-1], path)
            new = False
        else:
            print(f'{image} no tiene landmarks. Se generarán automaticamente.')
            landmarks = get_landmarks(os.path.join(path, image))
            if landmarks is None:
                print(f'No se pueden generar los landmarks de {image}.')
                continue
            new = True
            new_landmarks_file = chose_name(f'{landmarks_file_start}00.txt', path)
        c = Corrector(cv2.imread(os.path.join(path, image)), landmarks)
        print(f'Corrige landmarks de {image}...')
        update = c.show()
        cv2.destroyWindow(c.title)
        if update:
            d = dict(zip(points_interest_closed if 'M1' in new_landmarks_file else points_interest_opened, c.points.tolist()))

            if 'M1' in new_landmarks_file:
                d |= compute_distances(mesure_closed(dict(zip(points_interest_closed, c.points))))
            elif 'M2' in new_landmarks_file:
                d |= compute_distances(mesure_opened(dict(zip(points_interest_opened, c.points))))
            else:
                print(f'{new_landmarks_file} no contiene ni "M1" ni "M2" en el nombre. No se pueden calcular las medidas.', file=sys.stderr)

            with open(new_landmarks_file, 'w') as f:
                f.write(str(d).replace(", '", ",\n'").replace("{'", "{\n'").replace("}", "\n}").replace("': [", "':\t["))
            print(f'Puntos modificados guardados en {new_landmarks_file}')
        elif new:
            d = dict(zip(points_interest_closed if 'M1' in new_landmarks_file else points_interest_opened, c.points.tolist()))
            with open(new_landmarks_file, 'w') as f:
                f.write(str(d).replace(", '", ",\n'").replace("{'", "{\n'").replace("}", "\n}").replace("': [", "':\t["))
            print(f'Puntos generados guardados en {new_landmarks_file}')
        else:
            print(f'No se modificaron los puntos o no se quisieron guardar.')


if __name__ == '__main__':
    main(*sys.argv[1:2])
