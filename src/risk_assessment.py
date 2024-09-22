# src/risk_assessment.py

from tensorflow.keras.models import load_model
from preprocessing import preprocess_individual_patient
import os

def get_patient_risk_assessment(patient_id, medication_name):
    # Preprocess the patient's data with the specified medication
    features = preprocess_individual_patient(patient_id, medication_name)
    if features is None:
        return None

    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.h5')
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return None
    model = load_model(model_path)

    # Make a prediction
    risk_probability = model.predict(features)[0][0]

    # Interpret the risk level
    if risk_probability >= 0.7:
        risk_level = "High"
    elif risk_probability >= 0.4:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        'PatientID': patient_id,
        'RiskLevel': risk_level,
        'RiskProbability': risk_probability
    }
