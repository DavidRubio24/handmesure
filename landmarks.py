import numpy as np
from mediapipe.python.solutions.hands import Hands

from point_names import *
# from main import points_interest_closed, points_interest_opened

TIPS = [THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]
DIPS = [THUMB_IP, INDEX_FINGER_DIP, MIDDLE_FINGER_DIP, RING_FINGER_DIP, PINKY_DIP]


def get_landmarks(image_rgb: np.ndarray, closed: bool, detector=Hands(static_image_mode=True, max_num_hands=1)):
    results = detector.process(image_rgb)

    if results is None or results.multi_hand_landmarks is None:
        return None

    landmarks = np.array([(l.x, l.y) for l in results.multi_hand_landmarks[0].landmark])
    landmarks[:, 0] *= image_rgb.shape[1]
    landmarks[:, 1] *= image_rgb.shape[0]

    return get_landmarks_closed(image_rgb, landmarks) if closed else get_landmarks_opened(image_rgb, landmarks)


points_interest_opened = ['O_f1Tip', 'O_f1DistalR', 'O_f1DistalL', 'O_f2Tip', 'O_f2DistalR', 'O_f2DistalL',
                          'O_f2MedialR', 'O_f2MedialL', 'O_f3Tip', 'O_f3DistalR', 'O_f3DistalL', 'O_f3MedialR',
                          'O_f3MedialL', 'O_f4Tip', 'O_f4DistalR', 'O_f4DistalL', 'O_f4MedialR', 'O_f4MedialL',
                          'O_f5Tip', 'O_f5DistalR', 'O_f5DistalL', 'O_f5MedialR', 'O_f5MedialL']


def get_landmarks_opened(image, landmarks_mp: np.ndarray):
    landmarks_mp = np.round(landmarks_mp)
    landmarks_true = np.zeros((len(points_interest_opened), 2), int)
    # THUMB
    landmarks_true[0] = get_line_edge(image, landmarks_mp[THUMB_TIP], 2 * landmarks_mp[THUMB_TIP] - landmarks_mp[THUMB_IP])
    finger_direction = landmarks_mp[THUMB_TIP] - landmarks_mp[THUMB_MCP]
    finger_direction /= 3
    landmarks_true[1] = get_line_edge(image, landmarks_mp[THUMB_IP], direction=[-finger_direction[1], finger_direction[0]])
    landmarks_true[2] = get_line_edge(image, landmarks_mp[THUMB_IP], direction=[finger_direction[1], -finger_direction[0]])

    # INDEX
    landmarks_true[3] = get_line_edge(image, landmarks_mp[INDEX_FINGER_TIP], 2 * landmarks_mp[INDEX_FINGER_TIP] - landmarks_mp[INDEX_FINGER_PIP])
    finger_direction = landmarks_mp[INDEX_FINGER_TIP] - landmarks_mp[INDEX_FINGER_PIP]
    finger_direction /= 3
    landmarks_true[4] = get_line_edge(image, landmarks_mp[INDEX_FINGER_DIP], direction=[-finger_direction[1], finger_direction[0]])
    landmarks_true[5] = get_line_edge(image, landmarks_mp[INDEX_FINGER_DIP], direction=[finger_direction[1], -finger_direction[0]])
    finger_direction = landmarks_mp[INDEX_FINGER_DIP] - landmarks_mp[INDEX_FINGER_MCP]
    finger_direction /= 3
    landmarks_true[6] = get_line_edge(image, landmarks_mp[INDEX_FINGER_PIP], direction=[-finger_direction[1], finger_direction[0]])
    landmarks_true[7] = get_line_edge(image, landmarks_mp[INDEX_FINGER_PIP], direction=[finger_direction[1], -finger_direction[0]])

    # MIDDLE
    landmarks_true[8] = get_line_edge(image, landmarks_mp[MIDDLE_FINGER_TIP], 2 * landmarks_mp[MIDDLE_FINGER_TIP] - landmarks_mp[MIDDLE_FINGER_PIP])
    finger_direction = landmarks_mp[MIDDLE_FINGER_TIP] - landmarks_mp[MIDDLE_FINGER_PIP]
    finger_direction /= 3
    landmarks_true[9] = get_line_edge(image, landmarks_mp[MIDDLE_FINGER_DIP], direction=[-finger_direction[1], finger_direction[0]])
    landmarks_true[10] = get_line_edge(image, landmarks_mp[MIDDLE_FINGER_DIP], direction=[finger_direction[1], -finger_direction[0]])
    finger_direction = landmarks_mp[MIDDLE_FINGER_DIP] - landmarks_mp[MIDDLE_FINGER_MCP]
    finger_direction /= 3
    landmarks_true[11] = get_line_edge(image, landmarks_mp[MIDDLE_FINGER_PIP], direction=[-finger_direction[1], finger_direction[0]])
    landmarks_true[12] = get_line_edge(image, landmarks_mp[MIDDLE_FINGER_PIP], direction=[finger_direction[1], -finger_direction[0]])

    # RING
    landmarks_true[13] = get_line_edge(image, landmarks_mp[RING_FINGER_TIP], 2 * landmarks_mp[RING_FINGER_TIP] - landmarks_mp[RING_FINGER_PIP])
    finger_direction = landmarks_mp[RING_FINGER_TIP] - landmarks_mp[RING_FINGER_PIP]
    finger_direction /= 3
    landmarks_true[14] = get_line_edge(image, landmarks_mp[RING_FINGER_DIP], direction=[-finger_direction[1], finger_direction[0]])
    landmarks_true[15] = get_line_edge(image, landmarks_mp[RING_FINGER_DIP], direction=[finger_direction[1], -finger_direction[0]])
    finger_direction = landmarks_mp[RING_FINGER_DIP] - landmarks_mp[RING_FINGER_MCP]
    finger_direction /= 3
    landmarks_true[16] = get_line_edge(image, landmarks_mp[RING_FINGER_PIP], direction=[-finger_direction[1], finger_direction[0]])
    landmarks_true[17] = get_line_edge(image, landmarks_mp[RING_FINGER_PIP], direction=[finger_direction[1], -finger_direction[0]])

    # PINKY
    landmarks_true[18] = get_line_edge(image, landmarks_mp[PINKY_TIP], 2 * landmarks_mp[PINKY_TIP] - landmarks_mp[PINKY_PIP])
    finger_direction = landmarks_mp[PINKY_TIP] - landmarks_mp[PINKY_PIP]
    finger_direction /= 3
    landmarks_true[19] = get_line_edge(image, landmarks_mp[PINKY_DIP], direction=[-finger_direction[1], finger_direction[0]])
    landmarks_true[20] = get_line_edge(image, landmarks_mp[PINKY_DIP], direction=[finger_direction[1], -finger_direction[0]])
    finger_direction = landmarks_mp[PINKY_DIP] - landmarks_mp[PINKY_MCP]
    finger_direction /= 3
    landmarks_true[21] = get_line_edge(image, landmarks_mp[PINKY_PIP], direction=[-finger_direction[1], finger_direction[0]])
    landmarks_true[22] = get_line_edge(image, landmarks_mp[PINKY_PIP], direction=[finger_direction[1], -finger_direction[0]])

    return landmarks_true


points_interest_closed = ['C_f1Tip', 'C_f2Tip', 'C_f3Tip', 'C_f4Tip', 'C_f5Tip', 'C_f1BaseC', 'C_f2BaseC', 'C_f3BaseC',
                          'C_f4BaseC', 'C_f5BaseC', 'C_f1Defect', 'C_wristBaseC', 'C_palmBaseC', 'C_m1_2', 'C_m1_3']


def get_landmarks_closed(image, landmarks_mp: np.ndarray):
    landmarks_mp = np.round(landmarks_mp)
    landmarks_true = np.zeros((len(points_interest_closed), 2), int)
    # THUMB
    landmarks_true[0] = get_line_edge(image, landmarks_mp[THUMB_TIP], 2 * landmarks_mp[THUMB_TIP] - landmarks_mp[THUMB_IP])
    landmarks_true[5] = landmarks_mp[THUMB_MCP] * .95 + landmarks_mp[THUMB_CMC] * .05
    landmarks_true[10] = landmarks_mp[THUMB_MCP] * .7 + landmarks_mp[INDEX_FINGER_MCP] * .3

    # INDEX
    landmarks_true[1] = get_line_edge(image, landmarks_mp[INDEX_FINGER_TIP], 2 * landmarks_mp[INDEX_FINGER_TIP] - landmarks_mp[INDEX_FINGER_PIP])
    landmarks_true[6] = landmarks_mp[INDEX_FINGER_MCP] * (2/3) + landmarks_mp[INDEX_FINGER_PIP] / 3

    # MIDDLE
    landmarks_true[2] = get_line_edge(image, landmarks_mp[MIDDLE_FINGER_TIP], 2 * landmarks_mp[MIDDLE_FINGER_TIP] - landmarks_mp[MIDDLE_FINGER_PIP])
    landmarks_true[7] = landmarks_mp[MIDDLE_FINGER_MCP] * (2/3) + landmarks_mp[MIDDLE_FINGER_PIP] / 3

    # RING
    landmarks_true[3] = get_line_edge(image, landmarks_mp[RING_FINGER_TIP], 2 * landmarks_mp[RING_FINGER_TIP] - landmarks_mp[RING_FINGER_PIP])
    landmarks_true[8] = landmarks_mp[RING_FINGER_MCP] * (2/3) + landmarks_mp[RING_FINGER_PIP] / 3

    # PINKY
    landmarks_true[4] = get_line_edge(image, landmarks_mp[PINKY_TIP], 2 * landmarks_mp[PINKY_TIP] - landmarks_mp[PINKY_PIP])
    landmarks_true[9] = landmarks_mp[PINKY_MCP] * (2/3) + landmarks_mp[PINKY_PIP] / 3

    landmarks_true[11] = landmarks_mp[WRIST] * 1.1 - landmarks_mp[MIDDLE_FINGER_MCP] * .1
    landmarks_true[12] = landmarks_mp[WRIST]

    landmarks_true[13] = get_line_edge(image, landmarks_mp[INDEX_FINGER_MCP], direction=landmarks_mp[INDEX_FINGER_MCP]-landmarks_mp[MIDDLE_FINGER_MCP])
    landmarks_true[14] = get_line_edge(image, landmarks_mp[PINKY_MCP], direction=landmarks_mp[PINKY_MCP]-landmarks_mp[RING_FINGER_MCP])

    return landmarks_true


def get_line_edge(image, point1: np.ndarray, point2=None, direction=None):
    """Get the location of the edge of the hand in the continuation of the line between the first and second point."""
    if not 0 <= point1[0] < image.shape[1] or not 0 <= point1[1] < image.shape[0]:
        return point1

    point2 = point2 if point2 is not None else point1 + direction
    # Get the locations of the line that goes from the first to the second point.
    line_length = int(np.ceil(np.linalg.norm(point1 - point2)))
    line_locations = np.linspace(point1, point2, line_length, dtype=int)

    # Crop the indices to the image.
    line_locations = line_locations[(line_locations[:, 0] >= 0) & (line_locations[:, 0] < image.shape[1])]
    line_locations = line_locations[(line_locations[:, 1] >= 0) & (line_locations[:, 1] < image.shape[0])]

    # Get the color of each pixel in the line.
    line = np.array([image[y, x] for x, y in line_locations])

    # Find the index at which the color changes the most.
    change = line[1:].astype(int) - line[:-1]
    edge = np.argmax(np.linalg.norm(change, axis=1))
    edge_location = line_locations[edge]

    return edge_location
