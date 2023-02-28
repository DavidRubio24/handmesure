import os.path
import sys
import time

import cv2
import logging
import numpy as np
# There are additional imports in the function get_landmarks to avoid
# importing TensorFlow when not needed. TF is imported in model.py.

from constants import points_interest_closed, points_interest_opened
from correct import Corrector
from mesure import mesure_closed, mesure_opened, compute_distances

log = logging.getLogger(__name__); log.setLevel(logging.DEBUG)
formatter = logging.Formatter('\x1B[0;34m{asctime} {name}.{funcName}:{lineno} {levelname:5}\x1B[0m\t{message}', style='{')
log.addHandler(logging.StreamHandler()); log.handlers[-1].setFormatter(formatter)


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
    log.debug(f'Hands results: {results}')
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


def main(path=r'\\10.10.204.24\scan4d\TENDER\HANDS_CALIBRADAS', auto=False):
    """:param auto: If True, the landmarks will be generated without human correction."""
    if not os.path.isdir(path):
        print(f'Ruta inexistente ({path}).', file=sys.stderr)
        return

    images = [file for file in os.listdir(path) if file.lower().endswith(('.png', '.bmp')) and not file.lower().endswith('mesures.png')]

    print(f'{len(images)} imágenes en {os.path.abspath(path)}.')

    durations = []
    for image in images:
        start = time.time()
        # Search for landmarks file.
        landmarks_file_start = image[:-4] + '_lm'
        landmarks_files = sorted([file for file in os.listdir(path) if file.startswith(landmarks_file_start)])
        if landmarks_files:
            landmarks_file = os.path.join(path, landmarks_files[-1])  # Most recent landmarks file.
            with open(landmarks_file) as f:
                d = eval(f.read())
            landmarks = np.array([v for k, v in d.items() if k in points_interest_opened or k in points_interest_closed])
            print(f'Leyendo landmarks de {landmarks_file}.')
            new_landmarks_file = landmarks_file
            new = False
        else:
            print(f'{image} no tiene landmarks. Se generarán automaticamente.')
            landmarks = get_landmarks(os.path.join(path, image))
            if landmarks is None:
                print(f'No se pueden generar los landmarks de {image}.')
                continue
            new = True
            new_landmarks_file = os.path.join(path, f'{landmarks_file_start}.json')
        c = Corrector(os.path.join(path, image), landmarks)
        if not auto:
            print(f'Corrige landmarks de {image}...')
            updated = c.show()
            cv2.destroyWindow(c.title)
        if auto or new or updated:
            d = dict(zip(points_interest_closed if 'M1' in new_landmarks_file else points_interest_opened, c.points.tolist()))

            # If the date is in the filename, add it to the json file.
            if len(image) >= 13 and image[-13] == '.' and image[-12:-4].isdigit():
                date = image[-12:-4]
                d['capture_date'] = date[:4] + '-' + date[4:6] + '-' + date[6:]

            if 'M1' in new_landmarks_file:
                d |= compute_distances(mesure_closed(dict(zip(points_interest_closed, c.points))))
            elif 'M2' in new_landmarks_file:
                d |= compute_distances(mesure_opened(dict(zip(points_interest_opened, c.points))))
            else:
                print(f'{new_landmarks_file} no contiene ni "M1" ni "M2" en el nombre. No se pueden calcular las medidas.', file=sys.stderr)

            with open(new_landmarks_file, 'w') as f:
                # Write it as a well-formatted json file.
                f.write(str(d)
                        .replace(", '", ",\n'")
                        .replace("{'", "{\n'")
                        .replace("}", "\n}")
                        .replace("': [", "':\t[")
                        .replace("'", '"'))
            print(f'Puntos {"modificados" if updated else "generados"} guardados en {new_landmarks_file}')
        else:
            print(f'No se modificaron los puntos o no se quisieron guardar.')
        duration = int(time.time() - start)
        durations.append(duration)
        print(f'Tiempo empleado: {duration // 60: 2} min {duration % 60: 2} s.')

    print('Fin: todas las imágenes han sido corregidas.')
    print(f'Duraciones en segundos (media {np.mean(durations)} s):\n', durations)


if __name__ == '__main__':
    main(*sys.argv[1:2])
