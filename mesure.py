import os
import sys

import numpy as np


def mesure_closed(points: dict):
    points = {key: np.array(value, copy=False) for key, value in points.items()}

    # Compute the distance between the points in pixels.
    distance = {'handLength':          (points['C_f3Tip'],   points['C_wristBaseC']),
                'palmLength':          (points['C_f3BaseC'], points['C_palmBaseC']),
                'handThumbLength':     (points['C_f1Tip'],   points['C_f1BaseC']),
                'handIndexLength':     (points['C_f2Tip'],   points['C_f2BaseC']),
                'handMidLength':       (points['C_f3Tip'],   points['C_f3BaseC']),
                'handFourLength':      (points['C_f4Tip'],   points['C_f4BaseC']),
                'handLittleLength':    (points['C_f5Tip'],   points['C_f5BaseC']),
                # 'handBreadthMeta_C_m1_3-C_m1_2': (points['C_m1_2'], points['C_m1_3']),
                }

    # handLengthCrotch parallel to middle finger.
    direction = abs(points['C_f3Tip']) - abs(points['C_f3BaseC'])
    direction /= np.linalg.norm(direction)

    handLengthCrotch = np.dot(abs(points['C_f3Tip']) - abs(points['C_f1Defect']), direction) * direction + abs(points['C_f1Defect'])
    distance['handLengthCrotch'] = (handLengthCrotch, points['C_f1Defect'])
    if np.any(points['C_f3Tip'] < 0) or np.any(points['C_f3BaseC'] < 0) or np.any(points['C_f1Defect'] < 0):
        distance['handLengthCrotch'] = tuple(np.array(distance['handLengthCrotch']) * -1)

    # handBreadthMeta perpendicular to middle finger.
    # direction[:] = -direction[1], direction[0]
    # handBreadthMeta = np.dot(points['C_m1_2'] - points['C_m1_3'], direction) * direction + points['C_m1_3']
    # distance['handBreadthMeta_perpendicular_finger3'] = (handBreadthMeta, points['C_m1_3'])
    direction = abs(points['C_f3Tip']) - abs(points['C_wristBaseC'])
    direction /= np.linalg.norm(direction)
    direction[:] = -direction[1], direction[0]
    handBreadthMeta = np.dot(abs(points['C_m1_2']) - abs(points['C_m1_3']), direction) * direction + abs(points['C_m1_3'])
    distance['handBreadthMeta_perpendicular_hand'] = (handBreadthMeta, abs(points['C_m1_3']))
    if np.any(abs(points['C_m1_2']) < 0) or np.any(abs(points['C_m1_3']) < 0) or np.any(abs(points['C_m1_3']) < 0):
        distance['handBreadthMeta_perpendicular_hand'] = tuple(np.array(distance['handBreadthMeta_perpendicular_hand']) * -1)

    return distance


def mean_sign(x, y):
    """Return the mean of the absolute value of x and y with a negative sign if something is negative."""
    sign = 1 if np.all(x >= 0) and np.all(y >= 0) else -1
    return sign * np.mean([abs(x), abs(y)], axis=0)


def mesure_opened(points: dict):
    points = {key: np.array(value, copy=False) for key, value in points.items()}

    # Compute the distance between the points in pixels.
    distance = {'handThumbBreadth':        (points['O_f1DistalR'], points['O_f1DistalL']),
                'handIndexBreadthDistal':  (points['O_f2DistalR'], points['O_f2DistalL']),
                'handMidBreadthDistal':    (points['O_f3DistalR'], points['O_f3DistalL']),
                'handFourBreadthDistal':   (points['O_f4DistalR'], points['O_f4DistalL']),
                'handLittleBreadthDistal': (points['O_f5DistalR'], points['O_f5DistalL']),

                'handIndexBreadthProx':    (points['O_f2MedialR'], points['O_f2MedialL']),
                'handMidBreadthMid':       (points['O_f3MedialR'], points['O_f3MedialL']),
                'handFourBreadthMid':      (points['O_f4MedialR'], points['O_f4MedialL']),
                'handLittleBreadthMid':    (points['O_f5MedialR'], points['O_f5MedialL']),

                'handThumbLengthDistal':   (points['O_f1Tip'], mean_sign(points['O_f1DistalL'], points['O_f1DistalR'])),
                'handIndexLengthDistal':   (points['O_f2Tip'], mean_sign(points['O_f2DistalL'], points['O_f2DistalR'])),
                'handMidLengthDistal':     (points['O_f3Tip'], mean_sign(points['O_f3DistalL'], points['O_f3DistalR'])),
                'handFourLengthDistal':    (points['O_f4Tip'], mean_sign(points['O_f4DistalL'], points['O_f4DistalR'])),
                'handLittleLengthDistal':  (points['O_f5Tip'], mean_sign(points['O_f5DistalL'], points['O_f5DistalR'])),

                'handIndexLengthMid':     (mean_sign(points['O_f2DistalL'], points['O_f2DistalR']),
                                           mean_sign(points['O_f2MedialL'], points['O_f2MedialR'])),
                'handMidLengthMid':       (mean_sign(points['O_f3DistalL'], points['O_f3DistalR']),
                                           mean_sign(points['O_f3MedialL'], points['O_f3MedialR'])),
                'handFourLengthMid':      (mean_sign(points['O_f4DistalL'], points['O_f4DistalR']),
                                           mean_sign(points['O_f4MedialL'], points['O_f4MedialR'])),
                'handLittleLengthMid':    (mean_sign(points['O_f5DistalL'], points['O_f5DistalR']),
                                           mean_sign(points['O_f5MedialL'], points['O_f5MedialR'])),
                }

    return distance


def compute_distances(points: dict, factor: float = 138/1711.1):
    distances = {k: np.linalg.norm(abs(v[1]) - abs(v[0])) * factor * (1 if np.all(v[0] >= 0) and np.all(v[1] >= 0) else -1)
                 for k, v in points.items()}
    return distances


def to_list(points: dict):
    return {k: list(map(list, v)) for k, v in points.items()}


def main(path='./data'):
    if not os.path.isdir(path):
        print('Ruta inexistente.', file=sys.stderr)
        return

    landmarks = [file for file in os.listdir(path) if file.endswith('.txt') and '_lm' in file]

    for landmark in landmarks:
        d = eval(open(os.path.join(path, landmark)).read())
        if 'M1' in landmark:
            distances = to_list(mesure_closed(d))
        elif 'M2' in landmark:
            distances = to_list(mesure_opened(d))
        else:
            # TODO: The mesurement could perfectly be done type agnostic. .... Wait... Can it?
            print(f'{landmark} no contiene ni "M1" ni "M2" en el nombre.'
                  f'No se pueden calcular las distancias entre puntos.', file=sys.stderr)
            continue

        # Write the distances to a file.
        string = str(distances).replace("{'", "{\n'").replace("}", "\n}")
        string = string.replace(", '", ",\n'").replace("': ", "':\t")
        with open(os.path.join(path, landmark.replace('_lm', '_med')), 'w') as f:
            f.write(string)


if __name__ == '__main__':
    main(*sys.argv[1:])
