import os
import torch
from typing import Dict, Any
try:
    # When running as a package (e.g., from main.py)
    from .patient_risk_model import SimpleRiskPredictor
except ImportError:
    # When running directly (e.g., train_model.py)
    from patient_risk_model import SimpleRiskPredictor

class RiskPredictor:
    def __init__(self):
        self.model = SimpleRiskPredictor()
        # Dynamically resolve the path to the model file
        model_path = os.path.join(
            os.path.dirname(__file__), 'ml', 'simple_risk_model.pth'
        )
        try:
            print(f"Attempting to load model from: {model_path}")
            self.model.model.load_state_dict(torch.load(model_path))
            print("Loaded trained model successfully.")
        except FileNotFoundError:
            print(f"Model file not found at {model_path}. Please ensure the model exists.")
        except Exception as e:
            print(f"Error loading model: {e}")
        
    def predict(self, cursor, patient_id: str, new_med_id: int = None) -> Dict[str, Any]:
        """Make predictions for a patient"""
        result = self.model.predict(patient_id)
        
        # Convert to format expected by UI
        return {
            'ade_risk': result['risk_score'],
            'interaction_risk': result['risk_score'] * 0.8,  # Simplified approximation
            'overall_risk': result['risk_score'],
            'vital_risk': result['risk_score'] * 0.7,  # Simplified approximation
            'vital_warnings': []  # Simplified - remove complexity of vital warnings
        }
    
    def get_features(self, cursor, patient_id: str, new_med_id: int = None):
        """Keep this method for UI compatibility but simplify it"""
        return self.model._get_feature_importance(patient_id)
