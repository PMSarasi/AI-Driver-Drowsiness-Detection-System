import numpy as np
import cv2

def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_eye_landmarks(landmarks, indices):
    eye_coords = []
    for idx in indices:
        eye_coords.append([landmarks[idx].x, landmarks[idx].y])
    return np.array(eye_coords)

def draw_eye_landmarks(frame, eye_points, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    pixel_points = []
    for point in eye_points:
        x = int(point[0] * w)
        y = int(point[1] * h)
        pixel_points.append((x, y))
    for point in pixel_points:
        cv2.circle(frame, point, 2, color, -1)
    return frame

LEFT_EYE_SIMPLE = [33, 246, 161, 160, 159, 158]
RIGHT_EYE_SIMPLE = [362, 398, 384, 385, 386, 387]
