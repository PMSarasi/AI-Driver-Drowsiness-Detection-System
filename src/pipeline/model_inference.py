import tensorflow as tf
import cv2
import numpy as np
import os

class ModelInference:
    def __init__(self, models_path='models_saved'):
        self.cnn_model = None
        self.eye_model = None
        self.yawn_model = None
        self.models_path = models_path
        self.load_models()
    
    def load_models(self):
        """Load all three trained models"""
        try:
            cnn_path = os.path.join(self.models_path, 'cnn_ddd_quick.h5')
            self.cnn_model = tf.keras.models.load_model(cnn_path)
            print("✅ CNN Model loaded (96.25%)")
        except Exception as e:
            print(f"⚠️ CNN Model not loaded: {e}")
        
        try:
            eye_path = os.path.join(self.models_path, 'eye_model.h5')
            self.eye_model = tf.keras.models.load_model(eye_path)
            print("✅ Eye Model loaded (98.97%)")
        except Exception as e:
            print(f"⚠️ Eye Model not loaded: {e}")
        
        try:
            yawn_path = os.path.join(self.models_path, 'yawn_model_mouth_roi.h5')
            self.yawn_model = tf.keras.models.load_model(yawn_path)
            print("✅ Yawn Model loaded (99.31%)")
        except Exception as e:
            print(f"⚠️ Yawn Model not loaded: {e}")
    
    def preprocess_image(self, image):
        """Preprocess image for model inference"""
        if image is None:
            return None
        img = cv2.resize(image, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    
    def predict_cnn(self, face_roi):
        """Predict drowsiness from full face"""
        if self.cnn_model is None or face_roi is None:
            return 0.0
        
        img = self.preprocess_image(face_roi)
        if img is not None:
            pred = self.cnn_model.predict(img, verbose=0)[0]
            return float(pred[1])  # Drowsy probability
        return 0.0
    
    def predict_eye(self, eye_roi):
        """Predict eye state (open/closed)"""
        if self.eye_model is None or eye_roi is None:
            return 0.0
        
        img = self.preprocess_image(eye_roi)
        if img is not None:
            pred = self.eye_model.predict(img, verbose=0)[0]
            return float(pred[1])  # Closed probability
        return 0.0
    
    def predict_yawn(self, mouth_roi):
        """Predict yawning"""
        if self.yawn_model is None or mouth_roi is None:
            return 0.0
        
        img = self.preprocess_image(mouth_roi)
        if img is not None:
            pred = self.yawn_model.predict(img, verbose=0)[0]
            return float(pred[1])  # Yawn probability
        return 0.0
