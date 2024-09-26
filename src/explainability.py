import shap
from tensorflow.keras.models import load_model
from preprocessing import preprocess_individual_patient

def explain_patient_risk(patient_id, medication_name):
    features = preprocess_individual_patient(patient_id, medication_name)

    if features is None:
        print(f"No data found for patient {patient_id}.")
        return

    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.h5')
    model = load_model(model_path)

    # Initialize SHAP Kernel Explainer (or other explainers like DeepExplainer)
    explainer = shap.KernelExplainer(model.predict, features)
    shap_values = explainer.shap_values(features)

    # SHAP Force Plot for individual patient prediction
    shap.force_plot(explainer.expected_value, shap_values[0], features)
