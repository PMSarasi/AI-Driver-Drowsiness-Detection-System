import numpy as np

def calculate_mar(mouth_points):
    A = np.linalg.norm(mouth_points[2] - mouth_points[6])
    B = np.linalg.norm(mouth_points[0] - mouth_points[4])
    if B == 0:
        return 0
    mar = A / B
    return mar

def get_mouth_landmarks(landmarks, indices):
    mouth_coords = []
    for idx in indices:
        mouth_coords.append([landmarks[idx].x, landmarks[idx].y])
    return np.array(mouth_coords)

MOUTH_INDICES = [61, 185, 40, 39, 291, 409, 270, 269]
