# src/risk_assessment.py

import tensorflow as tf
from preprocessing import preprocess_individual_patient

# Load the trained model
model = tf.keras.models.load_model('models/model.h5')

def get_patient_risk_assessment(patient_id):
    # Preprocess data for the individual patient
    patient_data = preprocess_individual_patient(patient_id)
    
    # Drop 'PatientID' column
    patient_data = patient_data.drop(['PatientID'], axis=1)
    
    # Ensure the data has the correct format for the model
    patient_data = patient_data.astype('float32')
    
    # Make prediction
    risk_prob = model.predict(patient_data)
    risk = (risk_prob > 0.5).astype(int)
    
    print(f"\nPatient ID: {patient_id}")
    print(f"Predicted ADE Risk: {'High' if risk[0][0] == 1 else 'Low'}")
    print(f"Risk Probability: {risk_prob[0][0]:.2f}")
