import os
import sys
import sqlite3
import torch
import traceback
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from patient_risk_model import PatientRiskNet
from risk_predictor import RiskPredictor

def calculate_critical_risk(value: float, critical_min: float) -> float:
    """Calculate exponential risk increase for values below critical"""
    deviation = (critical_min - value) / critical_min
    return min(1.0, 0.5 + (deviation * 2))

def calculate_warning_risk(value: float, min_normal: float, critical_min: float) -> float:
    """Calculate linear risk increase for values in warning range"""
    range_size = min_normal - critical_min
    deviation = (min_normal - value) / range_size
    return min(0.5, deviation * 0.5)

def test_risk_components(risk_predictor, cursor, patient_id: str, med_id: int):
    """Test individual risk components for a single patient/medication pair"""
    features = risk_predictor.get_features(cursor, patient_id, med_id)
    vitals = features['vitals'][0].cpu().numpy()
    
    cursor.execute("SELECT name FROM Medications WHERE medication_id = ?", (med_id,))
    med_name = cursor.fetchone()['name']
    
    cursor.execute("SELECT * FROM Vital_Ranges")
    ranges = {row['vital_name']: dict(row) for row in cursor.fetchall()}
    
    print(f"\nDetailed Risk Analysis for Patient {patient_id} - {med_name}")
    print("\nVital Signs Analysis:")
    for i, name in enumerate(risk_predictor.vital_names):
        value = vitals[i]
        range_data = ranges[name]
        risk = 0.0
        
        if value <= float(range_data['critical_min']):
            risk = calculate_critical_risk(value, float(range_data['critical_min']))
        elif value < float(range_data['min_normal']):
            risk = calculate_warning_risk(value, float(range_data['min_normal']), float(range_data['critical_min']))
            
        print(f"{name}: {value:.1f} (Risk: {risk:.3f})")
        
    risk_factors = risk_predictor.get_med_specific_features(cursor, patient_id, med_id)
    print(f"\nADE Risk Score: {risk_factors['risk_factors']['ade_score']:.3f}")
    print(f"Interaction Risk Score: {risk_factors['risk_factors']['interaction_score']:.3f}")
    
    prediction = risk_predictor.predict(cursor, patient_id, med_id)
    print(f"\nFinal Risk Scores:")
    print(f"ADE Risk: {prediction['ade_risk']:.1%}")
    print(f"Interaction Risk: {prediction['interaction_risk']:.1%}")
    print(f"Overall Risk: {prediction['overall_risk']:.1%}")

def test_medication_predictions(cursor, risk_predictor):
    test_cases = [
        ("P3835", "Patient with ADE history"),
        ("P6964", "Patient with multiple medications"),
        ("P4251", "Patient with severe recent ADE")
    ]
    
    test_meds = [
        (3, "Metoprolol"),
        (9, "Digoxin"),
        (12, "Amiodarone")
    ]
    
    for patient_id, desc in test_cases:
        print(f"\nTesting {desc} ({patient_id})")
        for med_id, med_name in test_meds:
            pred = risk_predictor.predict(cursor, patient_id, med_id)
            print(f"{med_name}: ADE Risk={pred['ade_risk']:.2%}, "
                  f"Interaction Risk={pred['interaction_risk']:.2%}, "
                  f"Overall Risk={pred['overall_risk']:.2%}")

def train_risk_model():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai_cdss.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        print("Initializing risk predictor...")
        risk_predictor = RiskPredictor()
        
        print("Starting model training...")
        risk_predictor.train(cursor, batch_size=16, epochs=10)
        
        model_path = os.path.join(os.path.dirname(__file__), 'trained_models')
        os.makedirs(model_path, exist_ok=True)
        model_file = os.path.join(model_path, 'patient_risk_model.pth')
        risk_predictor.save(model_file)
        print(f"\nModel saved to: {model_file}")
        
        test_medication_predictions(cursor, risk_predictor)
        
        print("\nDetailed Component Analysis:")
        test_risk_components(risk_predictor, cursor, "P3835", 3)  # Test Metoprolol
        
    finally:
        conn.close()

if __name__ == "__main__":
    try:
        train_risk_model()
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        traceback.print_exc()