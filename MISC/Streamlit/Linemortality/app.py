import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model package
model_filename = 'mortality_prediction_model.joblib'
model_path = os.path.join(os.path.dirname(__file__), model_filename)

if not os.path.exists(model_path):
    st.error(f"⚠️ Model file '{model_filename}' not found! Ensure it's in the same directory as this script.")
    st.stop()

# Load model and metadata
model_package = joblib.load(model_path)
model = model_package['model']
feature_names = model_package['feature_names']
feature_ranges = model_package['feature_ranges']
risk_thresholds = model_package.get('risk_thresholds', [0.2, 0.4])

# Set realistic defaults for numerical features
realistic_defaults = {
    "age": 65,  # Average ICU patient age
    "heart_rate": 85,  # Normal HR in ICU patients
    "sbp": 120,  # Systolic blood pressure
    "dbp": 70,  # Diastolic blood pressure
    "mbp": 90,  # Mean arterial pressure
    "resp_rate": 18,  # Respiratory rate
    "temperature": 37.0,  # Normal body temperature
    "spo2": 97,  # Oxygen saturation
    "apsiii_score": 50,  # APS-III Severity Score (0-100)
    "sapsii_score": 40  # SAPS-II Score (0-163)
}

# UI Title
st.title('30-Day Mortality Prediction After Central Line Insertion')

st.write("""
### Enter Patient Information
Provide the required patient parameters to estimate mortality risk.
""")

# Arrange inputs in two columns
col1, col2 = st.columns(2)
input_data = {}

# Create input fields using a form
with st.form("input_form"):
    for i, feature in enumerate(feature_names):
        with col1 if i % 2 == 0 else col2:  # Distribute inputs across two columns
            if feature in ['cancer', 'liver_disease', 'cva', 'rrt', 'multiple_lines']:
                # Checkbox for binary conditions (Yes = 1, No = 0)
                input_data[feature] = int(st.checkbox(f'{feature.replace("_", " ").title()}'))
            else:
                # Use realistic defaults where applicable
                default_value = realistic_defaults.get(feature, feature_ranges[feature]['mean'])
                input_data[feature] = st.number_input(
                    f'{feature.replace("_", " ").title()}',
                    min_value=float(feature_ranges[feature]['min']),
                    max_value=float(feature_ranges[feature]['max']),
                    value=float(default_value),
                    step=0.01
                )

    # Submit button inside the form
    submit_button = st.form_submit_button("Predict Risk")

if submit_button:
    # Convert input into DataFrame (Raw Inputs, No Scaling)
    input_df = pd.DataFrame([input_data])

    # Make prediction using raw inputs
    probability = model.predict_proba(input_df)[0][1]

    # Define risk category based on thresholds
    if probability < risk_thresholds[0]:
        risk_category, color = 'Low', 'green'
    elif probability < risk_thresholds[1]:
        risk_category, color = 'Medium', 'orange'
    else:
        risk_category, color = 'High', 'red'

    # Display results
    st.subheader("Prediction Results")
    st.metric(label="30-Day Mortality Probability", value=f"{probability:.1%}")
    st.markdown(f"**Risk Category:** <span style='color:{color}; font-size:22px;'>{risk_category}</span>", unsafe_allow_html=True)

    # Display risk interpretation
    st.subheader("Risk Interpretation")
    if risk_category == 'Low':
        st.write("- Standard monitoring recommended")
        st.write("- Review risk factors for potential modification")
    elif risk_category == 'Medium':
        st.write("- Enhanced monitoring recommended")
        st.write("- Consider early intervention for modifiable risk factors")
        st.write("- Regular reassessment of clinical status")
    else:
        st.write("- Close monitoring required")
        st.write("- Immediate attention to modifiable risk factors")
        st.write("- Consider intensive care management")
        st.write("- Early goals of care discussion recommended")

# Sidebar Information
st.sidebar.subheader('Model Information')
st.sidebar.write(f"Model Performance (AUC): **{model_package['metrics']['auc']:.3f}**")
st.sidebar.write("**Risk Categories:**")
st.sidebar.write(f"- Low: < {risk_thresholds[0]*100:.0f}% mortality risk")
st.sidebar.write(f"- Medium: {risk_thresholds[0]*100:.0f}% - {risk_thresholds[1]*100:.0f}% mortality risk")
st.sidebar.write(f"- High: > {risk_thresholds[1]*100:.0f}% mortality risk")

