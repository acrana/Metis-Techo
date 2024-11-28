import os
import sys
import sqlite3
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
from patient_risk_model import PatientRiskNet  # Direct import
from risk_predictor import RiskPredictor

def test_medication_predictions(cursor, risk_predictor):
    test_cases = [
        ("P3835", "Patient with ADE history"),  # Current test case
        ("P6964", "Patient with multiple medications"),  # Has 10+ meds
        ("P4251", "Patient with severe recent ADE")   # Hepatitis case
    ]

    for patient_id, description in test_cases:
        print(f"\nTesting {description} ({patient_id}):")
        base_predictions = risk_predictor.predict(cursor, patient_id)
        print(f"Baseline risks - ADE: {base_predictions['ade_risk']:.2%}, Interactions: {base_predictions['interaction_risk']:.2%}")

        test_medications = [
            (3, "Metoprolol"),   # Common with interactions
            (9, "Digoxin"),      # High-risk cardiac med
            (17, "Vancomycin"),  # Antibiotic with monitoring
            (12, "Amiodarone"),  # Complex interaction profile
            (2, "Warfarin")      # High bleeding risk
        ]
        
        for med_id, med_name in test_medications:
            print(f"\n{med_name} (ID: {med_id}):")
            predictions = risk_predictor.predict(cursor, patient_id, med_id)
            print(f"ADE Risk: {predictions['ade_risk']:.2%}")
            print(f"Interaction Risk: {predictions['interaction_risk']:.2%}")
            print(f"Overall Risk: {predictions['overall_risk']:.2%}")
            print("Analysis:", risk_predictor.explain_prediction(cursor, patient_id, med_id))
        
        # Get detailed explanation
        print("\nDetailed Analysis:")
        explanation = risk_predictor.explain_prediction(cursor, patient_id, med_id)
        print(explanation)

def train_risk_model():
    # Get database path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    db_path = os.path.join(parent_dir, 'ai_cdss.db')
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Initialize risk predictor
        print("Initializing risk predictor...")
        risk_predictor = RiskPredictor()
        
        # Initialize PatientRiskNet with database connection
        patient_risk_net = PatientRiskNet(conn)
        
        print("Starting model training...")
        risk_predictor.train(cursor, batch_size=16, epochs=5)
        
        # Save the trained model
        model_path = os.path.join(current_dir, 'trained_models')
        os.makedirs(model_path, exist_ok=True)
        model_file = os.path.join(model_path, 'patient_risk_model.pth')
        risk_predictor.save(model_file)
        print(f"\nModel saved to: {model_file}")
        
        # Test the model
        print("\nTesting model on sample patients...")
        cursor.execute("SELECT patient_id, name FROM Patients LIMIT 5")
        test_patients = cursor.fetchall()
        
        for patient in test_patients:
            predictions = risk_predictor.predict(cursor, patient['patient_id'])
            print(f"\nPatient: {patient['name']} ({patient['patient_id']})")
            print(f"ADE Risk: {predictions['ade_risk']:.2%}")
            print(f"Interaction Risk: {predictions['interaction_risk']:.2%}")
            print(f"Overall Risk: {predictions['overall_risk']:.2%}")
        
        # Test medication-specific predictions
        test_medication_predictions(cursor, risk_predictor)
        
    finally:
        conn.close()

if __name__ == "__main__":
    try:
        train_risk_model()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()