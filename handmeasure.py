import os
import sys
import time

import cv2
import numpy as np

# There's an additional import:
# from landmarks import get_landmarks
# It uses mediapipe, so it's only imported if necessary.
from correct import Corrector
from mesure import mesure_closed, mesure_opened, compute_distances

points_interest_closed = ['C_f1Tip', 'C_f2Tip', 'C_f3Tip', 'C_f4Tip', 'C_f5Tip', 'C_f1BaseC', 'C_f2BaseC', 'C_f3BaseC',
                          'C_f4BaseC', 'C_f5BaseC', 'C_f1Defect', 'C_wristBaseC', 'C_palmBaseC', 'C_m1_2', 'C_m1_3']

points_interest_opened = ['O_f1Tip', 'O_f1DistalR', 'O_f1DistalL', 'O_f2Tip', 'O_f2DistalR', 'O_f2DistalL',
                          'O_f2MedialR', 'O_f2MedialL', 'O_f3Tip', 'O_f3DistalR', 'O_f3DistalL', 'O_f3MedialR',
                          'O_f3MedialL', 'O_f4Tip', 'O_f4DistalR', 'O_f4DistalL', 'O_f4MedialR', 'O_f4MedialL',
                          'O_f5Tip', 'O_f5DistalR', 'O_f5DistalL', 'O_f5MedialR', 'O_f5MedialL']


def landmarks_from_file(file, closed=True):
    """Return the landmarks from the most up to date json file or None if there isn't."""
    path, name = os.path.split(file)
    basename, extension = os.path.splitext(name)

    landmarks_files = sorted([f for f in os.listdir(path or '.') if f.startswith(basename) and f.endswith('.json')])

    if not landmarks_files:
        return None

    print(f'Cargando puntos de {landmarks_files[-1]}...')

    with open(os.path.join(path, landmarks_files[-1]), 'r') as json_file:
        landmarks_dict = eval(json_file.read())

    points_interest = points_interest_closed if closed else points_interest_opened

    return np.array([landmarks_dict[point] for point in points_interest])


def main(path=r'\\10.10.204.24\scan4d\TENDER\HANDS\02_HANDS_CALIBRADAS/', auto=False):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png') and 'measure' not in f]
    durations = []
    for file in files:
        start = time.time()
        new = updated = False
        if 'close' in file or 'M1' in file.upper():
            closed = True
        elif 'open' in file or 'M2' in file.upper():
            closed = False
        else:
            print(f'{file} no es ni abierto ni cerrado. Se ignora.')
            continue

        landmarks = landmarks_from_file(file, closed)
        if landmarks is None:
            print(f'{file} no tiene landmarks. Se generarán automaticamente.')
            from landmarks import get_landmarks
            image = cv2.imread(file)
            if image is None:
                print(f'No se puede leer {file}.')
                continue
            image_rgb = image[..., ::-1]
            landmarks = get_landmarks(image_rgb, closed)
            new = True
            if landmarks is None:
                print(f'No se puede detectar la mano en {file}.')
                continue
            landmarks = landmarks + .0001

        c = Corrector(file, landmarks)

        if not auto:
            print(f'Corrige landmarks de {file}...')
            updated = c.show()
            cv2.destroyWindow(c.title)

        new_landmarks_file = file[:-4] + '.json'

        if auto or updated or new:
            d = dict(zip(points_interest_closed if 'M1' in new_landmarks_file or 'close' in new_landmarks_file else points_interest_opened, c.points.tolist()))

            # If the date is in the filename, add it to the json file.
            if len(file) >= 13 and file[-13] == '.' and file[-12:-4].isdigit():
                date = file[-12:-4]
                d['capture_date'] = date[:4] + '-' + date[4:6] + '-' + date[6:]

            if 'M1' in new_landmarks_file or 'close' in new_landmarks_file:
                d |= compute_distances(mesure_closed(dict(zip(points_interest_closed, c.points))))
            elif 'M2' in new_landmarks_file or 'open' in new_landmarks_file:
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
    main(*sys.argv[1:])
