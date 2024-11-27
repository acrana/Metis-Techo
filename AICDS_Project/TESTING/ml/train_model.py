# train_model.py
import os
import sys
import sqlite3
from risk_predictor import RiskPredictor

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
        # Initialize and train model
        print("Initializing risk predictor...")
        risk_predictor = RiskPredictor()
        
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
        
        # Get detailed analysis for first patient
        print("\nDetailed Analysis for First Patient:")
        first_patient = test_patients[0]
        explanation = risk_predictor.explain_prediction(cursor, first_patient['patient_id'])
        print(explanation)
    
    finally:
        conn.close()

if __name__ == "__main__":
    try:
        train_risk_model()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()