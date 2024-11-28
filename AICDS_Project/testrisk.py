import sys
import os
import sqlite3
import torch

# Add the parent directory of the 'ml' module to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from risk_predictor import RiskPredictor

def test_risk_predictor():
    # Connect to the test database
    db_path = "C:/Users/acran/OneDrive/Desktop/Projects/GPT/AI CDSS/TESTING - Copy - Copy/ai_cdss.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Initialize the RiskPredictor
    predictor = RiskPredictor()

    # Test prediction
    patient_id = 'test_patient'
    new_med_id = 123  # Example medication ID
    prediction = predictor.predict(cursor, patient_id, new_med_id)
    print(f"Prediction for patient {patient_id} with new medication {new_med_id}: {prediction}")

    # Test training
    predictor.train(cursor, batch_size=2, epochs=1)

    # Test explanation
    explanation = predictor.explain_prediction(cursor, patient_id, new_med_id)
    print(f"Explanation for patient {patient_id} with medication {new_med_id}: {explanation}")

    # Save and load model
    model_path = 'trained_models/risk_model.pth'
    predictor.save(model_path)
    predictor.load(model_path)

    # Close the database connection
    conn.close()

if __name__ == '__main__':
    test_risk_predictor()
