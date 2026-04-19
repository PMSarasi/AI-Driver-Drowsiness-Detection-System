import cv2
import numpy as np

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

def calculate_head_pose(frame, landmarks):
    h, w = frame.shape[:2]
    focal = w
    center = (w/2, h/2)
    cam_matrix = np.array([
        [focal, 0, center[0]],
        [0, focal, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1))
    
    points_2d = []
    for idx in LANDMARK_INDICES:
        lm = landmarks[idx]
        points_2d.append([lm.x * w, lm.y * h])
    points_2d = np.array(points_2d, dtype=np.float64)
    
    success, rot_vec, trans_vec = cv2.solvePnP(
        MODEL_POINTS, points_2d, cam_matrix, dist_coeffs
    )
    
    if not success:
        return 0, 0, 0
    
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    pose_mat = cv2.hconcat((rot_mat, trans_vec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
    
    pitch = euler[0][0]
    yaw = euler[1][0]
    roll = euler[2][0]
    
    return pitch, yaw, roll
