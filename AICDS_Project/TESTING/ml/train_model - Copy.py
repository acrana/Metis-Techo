# train_model.py
import os
import sys
import sqlite3
from risk_predictor import RiskPredictor

def test_medication_predictions(cursor, risk_predictor):
    # Test patient with known ADE history
    print("\nTesting Medication-Specific Predictions:")
    test_patient = "P6964"  # Sarah Nelson
    
    # First test without specific medication
    print(f"\nBaseline prediction for patient {test_patient}:")
    base_predictions = risk_predictor.predict(cursor, test_patient)
    print(f"ADE Risk: {base_predictions['ade_risk']:.2%}")
    print(f"Interaction Risk: {base_predictions['interaction_risk']:.2%}")
    print(f"Overall Risk: {base_predictions['overall_risk']:.2%}")
    
    # Test medications
    test_medications = [
        (3, "Metoprolol"),  # Test with a beta blocker
        (9, "Digoxin"),     # Test with a cardiac medication
        (17, "Vancomycin")  # Test with an antibiotic
    ]
    
    for med_id, med_name in test_medications:
        print(f"\nPrediction for {med_name} (ID: {med_id}):")
        predictions = risk_predictor.predict(cursor, test_patient, med_id)
        print(f"ADE Risk: {predictions['ade_risk']:.2%}")
        print(f"Interaction Risk: {predictions['interaction_risk']:.2%}")
        print(f"Overall Risk: {predictions['overall_risk']:.2%}")
        
        # Get detailed explanation
        print("\nDetailed Analysis:")
        explanation = risk_predictor.explain_prediction(cursor, test_patient, med_id)
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