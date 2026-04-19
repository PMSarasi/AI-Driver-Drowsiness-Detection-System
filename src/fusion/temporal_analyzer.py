from collections import deque
import numpy as np
import time

class TemporalAnalyzer:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.ear_history = deque(maxlen=window_size)
        self.mar_history = deque(maxlen=window_size)
        self.cnn_history = deque(maxlen=window_size)
        self.risk_history = deque(maxlen=100)
        
        # Timers for temporal events
        self.eyes_closed_start = None
        self.yawn_start = None
        self.drowsy_start = None
        
        # Thresholds
        self.EYE_CLOSED_THRESHOLD = 0.3  # 30% confidence = closed
        self.YAWN_THRESHOLD = 0.5  # 50% confidence = yawning
        self.DROWSY_THRESHOLD = 0.6  # 60% risk = drowsy
        
    def update(self, ear, mar, cnn_prob):
        """Update history buffers"""
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        self.cnn_history.append(cnn_prob)
        
        # Calculate smoothed values
        smooth_ear = np.mean(self.ear_history) if self.ear_history else 0
        smooth_mar = np.mean(self.mar_history) if self.mar_history else 0
        smooth_cnn = np.mean(self.cnn_history) if self.cnn_history else 0
        
        return smooth_ear, smooth_mar, smooth_cnn
    
    def detect_eye_closure_duration(self, eye_closed_prob):
        """Track how long eyes have been closed"""
        if eye_closed_prob > self.EYE_CLOSED_THRESHOLD:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = time.time()
            return time.time() - self.eyes_closed_start
        else:
            self.eyes_closed_start = None
            return 0
    
    def detect_yawn_duration(self, yawn_prob):
        """Track how long yawning has been detected"""
        if yawn_prob > self.YAWN_THRESHOLD:
            if self.yawn_start is None:
                self.yawn_start = time.time()
            return time.time() - self.yawn_start
        else:
            self.yawn_start = None
            return 0
    
    def detect_drowsy_duration(self, risk_score):
        """Track how long driver has been drowsy"""
        if risk_score > 60:
            if self.drowsy_start is None:
                self.drowsy_start = time.time()
            return time.time() - self.drowsy_start
        else:
            self.drowsy_start = None
            return 0
    
    def get_temporal_alerts(self, eye_closed_duration, yawn_duration, drowsy_duration):
        """Generate alerts based on temporal patterns"""
        alerts = []
        
        if eye_closed_duration > 2.0:
            alerts.append(f"EYES CLOSED {eye_closed_duration:.1f}s")
        
        if yawn_duration > 1.5:
            alerts.append(f"YAWNING {yawn_duration:.1f}s")
        
        if drowsy_duration > 3.0:
            alerts.append(f"DROWSY {drowsy_duration:.1f}s")
        
        return alerts
