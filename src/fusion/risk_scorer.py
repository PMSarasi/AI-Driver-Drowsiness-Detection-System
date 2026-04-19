import numpy as np

class RiskScorer:
    def __init__(self):
        # Weights for fusion
        self.CNN_WEIGHT = 0.35
        self.EAR_WEIGHT = 0.35
        self.MAR_WEIGHT = 0.30
        
        # Thresholds
        self.EAR_THRESHOLD = 0.2
        self.MAR_THRESHOLD = 0.6
        
    def calculate_ear_risk(self, ear):
        """Convert EAR to risk score (0-100)"""
        if ear < 0.15:
            return 100
        elif ear < 0.18:
            return 80
        elif ear < 0.20:
            return 60
        elif ear < 0.22:
            return 40
        elif ear < 0.25:
            return 20
        else:
            return 0
    
    def calculate_mar_risk(self, mar):
        """Convert MAR to risk score (0-100)"""
        if mar > 0.75:
            return 100
        elif mar > 0.65:
            return 80
        elif mar > 0.55:
            return 50
        elif mar > 0.45:
            return 20
        else:
            return 0
    
    def calculate_fusion_risk(self, cnn_prob, ear, mar):
        """
        Fusion risk scoring with weights:
        - CNN: 35%
        - EAR: 35%
        - MAR: 30%
        """
        # CNN contribution (already 0-1 probability)
        cnn_risk = cnn_prob * 100
        
        # EAR contribution
        ear_risk = self.calculate_ear_risk(ear)
        
        # MAR contribution
        mar_risk = self.calculate_mar_risk(mar)
        
        # Weighted fusion
        fusion_risk = (
            self.CNN_WEIGHT * cnn_risk +
            self.EAR_WEIGHT * ear_risk +
            self.MAR_WEIGHT * mar_risk
        )
        
        return min(100, fusion_risk), cnn_risk, ear_risk, mar_risk
    
    def get_status(self, risk_score):
        """Get status based on risk score"""
        if risk_score < 30:
            return "SAFE", (0, 255, 0)
        elif risk_score < 50:
            return "CAUTION", (0, 255, 255)
        elif risk_score < 70:
            return "WARNING", (0, 165, 255)
        else:
            return "DANGER!", (0, 0, 255)
