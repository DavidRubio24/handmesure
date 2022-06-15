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
                }

    # handLengthCrotch parallel to middle finger.
    direction = points['C_f3Tip'] - points['C_f3BaseC']
    direction /= np.linalg.norm(direction)

    handLengthCrotch = np.dot(points['C_f3Tip'] - points['C_f1Defect'], direction) * direction + points['C_f1Defect']
    distance['handLengthCrotch'] = (handLengthCrotch, points['C_f1Defect'])

    # handBreadthMeta perpendicular to middle finger.
    direction[:] = -direction[1], direction[0]
    handBreadthMeta = np.dot(points['C_m1_2'] - points['C_m1_3'], direction) * direction + points['C_m1_3']
    distance['handBreadthMeta'] = (handBreadthMeta, points['C_m1_3'])

    return distance


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

                'handThumbLengthDistal':   (points['O_f1Tip'], np.mean([points['O_f1DistalL'],
                                                                        points['O_f1DistalR']], axis=0)),
                'handIndexLengthDistal':   (points['O_f2Tip'], np.mean([points['O_f2DistalL'],
                                                                        points['O_f2DistalR']], axis=0)),
                'handMidLengthDistal':     (points['O_f3Tip'], np.mean([points['O_f3DistalL'],
                                                                        points['O_f3DistalR']], axis=0)),
                'handFourLengthDistal':    (points['O_f4Tip'], np.mean([points['O_f4DistalL'],
                                                                        points['O_f4DistalR']], axis=0)),
                'handLittleLengthDistal':  (points['O_f5Tip'], np.mean([points['O_f5DistalL'],
                                                                        points['O_f5DistalR']], axis=0)),

                'handIndexLengthMid':     (np.mean([points['O_f2DistalL'], points['O_f2DistalR']], axis=0),
                                           np.mean([points['O_f2MedialL'], points['O_f2MedialR']], axis=0)),
                'handMidLengthMid':       (np.mean([points['O_f3DistalL'], points['O_f3DistalR']], axis=0),
                                           np.mean([points['O_f3MedialL'], points['O_f3MedialR']], axis=0)),
                'handFourLengthMid':      (np.mean([points['O_f4DistalL'], points['O_f4DistalR']], axis=0),
                                           np.mean([points['O_f4MedialL'], points['O_f4MedialR']], axis=0)),
                'handLittleLengthMid':    (np.mean([points['O_f5DistalL'], points['O_f5DistalR']], axis=0),
                                           np.mean([points['O_f5MedialL'], points['O_f5MedialR']], axis=0)),
                }

    return distance


def compute_distances(points: dict, factor: float = 0.08):
    distances = {k: np.linalg.norm(v[1] - v[0]) * factor for k, v in points.items()}
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
            # TODO: The mesurement could perfectly be done type agnostic.
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
