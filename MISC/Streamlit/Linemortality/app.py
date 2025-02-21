import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model package
try:
    model_package = joblib.load('mortality_prediction_model.joblib')
except FileNotFoundError:
    st.error("Model file 'mortality_prediction_model.joblib' not found. Please ensure it’s in the correct directory.")
    st.stop()

st.title('30-Day Mortality Prediction After Central Line Insertion')

# Use feature_ranges instead of clinical_ranges for realistic bounds
ranges = model_package['feature_ranges']  # Actual data ranges from training

# Patient Information
st.subheader('Patient Information')
age = st.number_input(
    'Age (years)', 
    min_value=float(ranges['age']['min']),
    max_value=float(ranges['age']['max']),
    value=65.0
)

# Clinical Scores
st.subheader('Clinical Scores')
apsiii_score = st.number_input(
    'APSIII Score', 
    min_value=float(ranges['apsiii_score']['min']),
    max_value=float(ranges['apsiii_score']['max']),
    value=50.0
)
sapsii_score = st.number_input(
    'SAPSII Score',
    min_value=float(ranges['sapsii_score']['min']),
    max_value=float(ranges['sapsii_score']['max']),
    value=40.0
)

# Laboratory Values
st.subheader('Laboratory Values')
col1, col2 = st.columns(2)
with col1:
    bicarbonate_mean = st.number_input(
        'Bicarbonate (mEq/L)',
        min_value=float(ranges['bicarbonate_mean']['min']),
        max_value=float(ranges['bicarbonate_mean']['max']),
        value=24.0
    )
    sodium_mean = st.number_input(
        'Sodium (mEq/L)',
        min_value=float(ranges['sodium_mean']['min']),
        max_value=float(ranges['sodium_mean']['max']),
        value=140.0
    )
    aniongap_mean = st.number_input(
        'Anion Gap (mEq/L)',
        min_value=float(ranges['aniongap_mean']['min']),
        max_value=float(ranges['aniongap_mean']['max']),
        value=7.0
    )
with col2:
    glucose_mean = st.number_input(
        'Glucose (mg/dL)',
        min_value=float(ranges['glucose_mean']['min']),
        max_value=float(ranges['glucose_mean']['max']),
        value=100.0
    )
    platelet_mean = st.number_input(
        'Platelet Count (K/uL)',
        min_value=float(ranges['platelet_mean']['min']),
        max_value=float(ranges['platelet_mean']['max']),
        value=200.0
    )

# Vital Signs
st.subheader('Vital Signs')
col1, col2 = st.columns(2)
with col1:
    temperature_mean = st.number_input(
        'Temperature (°C)',
        min_value=float(ranges['temperature_mean']['min']),
        max_value=float(ranges['temperature_mean']['max']),
        value=37.0
    )
    sbp_mean = st.number_input(
        'Systolic Blood Pressure (mmHg)',
        min_value=float(ranges['sbp_mean']['min']),
        max_value=float(ranges['sbp_mean']['max']),
        value=110.0
    )
with col2:
    resp_rate_mean = st.number_input(
        'Respiratory Rate (breaths/min)',
        min_value=float(ranges['resp_rate_mean']['min']),
        max_value=float(ranges['resp_rate_mean']['max']),
        value=16.0
    )

# Binary Features
st.subheader('Other Clinical Factors')
col1, col2 = st.columns(2)
with col1:
    multiple_lines = st.checkbox('Multiple Central Lines')
    cancer = st.checkbox('Active Cancer')
with col2:
    diabetes = st.checkbox('Diabetes')
    liver_disease = st.checkbox('Liver Disease')

# Prediction Button
if st.button('Predict Mortality Risk'):
    try:
        # Create input data in the exact order of feature_names
        input_data = pd.DataFrame([[
            apsiii_score, sapsii_score, bicarbonate_mean, int(multiple_lines), aniongap_mean,
            int(cancer), temperature_mean, platelet_mean, sodium_mean, glucose_mean,
            int(diabetes), age, int(liver_disease), sbp_mean, resp_rate_mean
        ]], columns=model_package['feature_names'])
        
        # Make prediction
        prediction_prob = model_package['model'].predict_proba(input_data)[0][1]
        
        # Determine risk level based on thresholds from model_package
        if prediction_prob < 0.2:
            risk_level = 'Low'
            color = 'green'
        elif prediction_prob < 0.4:
            risk_level = 'Medium'
            color = 'orange'
        else:
            risk_level = 'High'
            color = 'red'
        
        # Display results
        st.header('Prediction Results')
        st.markdown(f'**Risk Level:** :{color}[{risk_level}]')
        st.markdown(f'**30-day Mortality Probability:** {prediction_prob:.1%}')
        
        # Interpretation
        st.subheader('Risk Level Interpretation')
        if risk_level == 'Low':
            st.write('Mortality risk is relatively low. Consider standard monitoring protocols.')
        elif risk_level == 'Medium':
            st.write('Elevated mortality risk. Consider increased monitoring and preventive measures.')
        else:
            st.write('High mortality risk. Consider intensive monitoring and preventive interventions.')
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Sidebar with Model Info
st.sidebar.markdown('### Model Information')
st.sidebar.write(f"AUC: {model_package['metrics']['auc']:.3f}")
st.sidebar.write(f"Brier Score: {model_package['metrics']['brier']:.3f}")
st.sidebar.write(f"Number of Features: {len(model_package['feature_names'])}")
