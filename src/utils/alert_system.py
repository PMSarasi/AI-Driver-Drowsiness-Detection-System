import cv2
import numpy as np
import time
import threading
import os

class AlertSystem:
    def __init__(self):
        self.alert_active = False
        self.alert_start_time = 0
        self.alert_cooldown = 3.0  # seconds between alerts
        self.last_alert_time = 0
        self.flash_alpha = 0.0
        
        # Try to load sound
        self.sound_enabled = False
        try:
            from playsound import playsound
            self.playsound = playsound
            self.sound_enabled = True
        except:
            print("Sound alerts disabled (playsound not available)")
    
    def trigger_alert(self, risk_score, status):
        """Trigger alert based on risk level"""
        current_time = time.time()
        
        if risk_score >= 70:
            if not self.alert_active:
                self.alert_active = True
                self.alert_start_time = current_time
                self.flash_alpha = 0.7
                
                # Play sound for danger
                if self.sound_enabled and (current_time - self.last_alert_time) > self.alert_cooldown:
                    self._play_alert_sound()
                    self.last_alert_time = current_time
            else:
                # Pulse the flash effect
                elapsed = current_time - self.alert_start_time
                self.flash_alpha = 0.5 + 0.3 * np.sin(elapsed * 10)
        elif risk_score >= 50:
            self.alert_active = True
            self.flash_alpha = 0.3
        else:
            self.alert_active = False
            self.flash_alpha = 0.0
    
    def _play_alert_sound(self):
        """Play alert sound in separate thread"""
        def play():
            try:
                # Use system beep as fallback
                import winsound
                winsound.Beep(1000, 300)
            except:
                pass
        threading.Thread(target=play, daemon=True).start()
    
    def apply_visual_alert(self, frame):
        """Apply visual alert overlay to frame"""
        if self.alert_active and self.flash_alpha > 0:
            h, w = frame.shape[:2]
            overlay = frame.copy()
            
            if self.flash_alpha > 0.5:
                # Red flash for danger
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            else:
                # Yellow flash for warning
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 255), -1)
            
            frame = cv2.addWeighted(overlay, self.flash_alpha * 0.3, frame, 1 - self.flash_alpha * 0.3, 0)
        
        return frame
    
    def draw_alert_text(self, frame, status, color):
        """Draw alert text on frame"""
        if self.alert_active:
            h, w = frame.shape[:2]
            
            # Alert banner at top
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"⚠️ {status} ⚠️", (w//2 - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame
