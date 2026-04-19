import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect(self, frame):
        """Detect face and return landmarks"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None
    
    def get_face_roi(self, frame, landmarks, padding=30):
        """Extract face region from landmarks"""
        h, w = frame.shape[:2]
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        x1 = max(0, int(min(x_coords) * w) - padding)
        y1 = max(0, int(min(y_coords) * h) - padding)
        x2 = min(w, int(max(x_coords) * w) + padding)
        y2 = min(h, int(max(y_coords) * h) + padding)
        
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
        return None, None
    
    def get_eye_roi(self, frame, landmarks, eye_indices):
        """Extract eye region"""
        h, w = frame.shape[:2]
        points = []
        for idx in eye_indices:
            points.append([landmarks[idx].x, landmarks[idx].y])
        points = np.array(points)
        
        x1 = max(0, int(np.min(points[:, 0]) * w) - 20)
        y1 = max(0, int(np.min(points[:, 1]) * h) - 20)
        x2 = min(w, int(np.max(points[:, 0]) * w) + 20)
        y2 = min(h, int(np.max(points[:, 1]) * h) + 20)
        
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2]
        return None
    
    def get_mouth_roi(self, frame, landmarks, mouth_indices):
        """Extract mouth region"""
        h, w = frame.shape[:2]
        points = []
        for idx in mouth_indices:
            points.append([landmarks[idx].x, landmarks[idx].y])
        points = np.array(points)
        
        x1 = max(0, int(np.min(points[:, 0]) * w) - 30)
        y1 = max(0, int(np.min(points[:, 1]) * h) - 20)
        x2 = min(w, int(np.max(points[:, 0]) * w) + 30)
        y2 = min(h, int(np.max(points[:, 1]) * h) + 20)
        
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2]
        return None
    
    def draw_landmarks(self, frame, landmarks):
        """Draw face mesh on frame"""
        # Simplified drawing - just key points
        h, w = frame.shape[:2]
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame
    
    def close(self):
        self.face_mesh.close()
