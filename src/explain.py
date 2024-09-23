
# Not implemented***************
# src/explainability.py

import os
import shap
import joblib
from tensorflow.keras.models import load_model
from preprocessing import preprocess_data  # Or use preprocess_individual_patient for single-patient data

def explain_model_with_shap():
    """
    Generate SHAP explanations for the trained model on the test data.
    """
    # Preprocess test data (or load preprocessed test data)
    data = preprocess_data()

    # Separate features and target
    X = data.drop(['Had_ADE'], axis=1)
    y = data['Had_ADE']

    # Load the scaler
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.joblib')
    scaler = joblib.load(scaler_path)

    # Scale the features
    X_scaled = scaler.transform(X)

    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.h5')
    model = load_model(model_path)

    # Initialize SHAP explainer
    explainer = shap.KernelExplainer(model.predict, X_scaled)

    # Generate SHAP values for the test data
    shap_values = explainer.shap_values(X_scaled)

    # SHAP summary plot to visualize feature importance
    shap.summary_plot(shap_values, X_scaled, feature_names=X.columns)

def explain_individual_patient_with_shap(patient_id, medication_name):
    """
    Generate SHAP explanations for a single patient.
    """
    from preprocessing import preprocess_individual_patient

    # Preprocess individual patient data with the new medication
    features = preprocess_individual_patient(patient_id, medication_name)

    if features is None:
        print(f"No data found for patient {patient_id}.")
        return

    # Load the scaler
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.joblib')
    scaler = joblib.load(scaler_path)
    features_scaled = scaler.transform(features)

    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.h5')
    model = load_model(model_path)

    # Initialize SHAP explainer
    explainer = shap.KernelExplainer(model.predict, features_scaled)

    # Generate SHAP values for the individual patient
    shap_values = explainer.shap_values(features_scaled)

    # SHAP force plot for detailed explanation of individual prediction
    shap.force_plot(explainer.expected_value, shap_values[0], features_scaled)

