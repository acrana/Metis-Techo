import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Get the absolute path to the directory containing app.py
dirname = os.path.dirname(__file__)
# Construct the full path to the model file
model_path = os.path.join(dirname, 'mortality_prediction_model.joblib')

# Load the saved model package
model_package = joblib.load(model_path)

st.title('30-Day Mortality Prediction After Central Line Insertion')

# Create sections for different types of inputs
st.subheader('Patient Information')
age = st.number_input(
    'Age (years)', 
    min_value=float(model_package['clinical_ranges']['age']['min']),
    max_value=float(model_package['clinical_ranges']['age']['max']),
    value=65.0,
    step=1.0
)

# Clinical Scores
st.subheader('Clinical Scores')
apsiii = st.number_input(
    'APSIII Score', 
    min_value=float(model_package['clinical_ranges']['apsiii_score']['min']),
    max_value=float(model_package['clinical_ranges']['apsiii_score']['max']),
    value=50.0,
    step=1.0
)
sapsii = st.number_input(
    'SAPSII Score',
    min_value=float(model_package['clinical_ranges']['sapsii_score']['min']),
    max_value=float(model_package['clinical_ranges']['sapsii_score']['max']),
    value=40.0,
    step=1.0
)

# Laboratory Values
st.subheader('Laboratory Values')
col1, col2 = st.columns(2)
with col1:
   bicarbonate = st.number_input(
       'Bicarbonate (mEq/L)',
       min_value=model_package['clinical_ranges']['bicarbonate_mean']['min'],
       max_value=model_package['clinical_ranges']['bicarbonate_mean']['max'],
       value=24.0
   )
   sodium = st.number_input(
       'Sodium (mEq/L)',
       min_value=model_package['clinical_ranges']['sodium_mean']['min'],
       max_value=model_package['clinical_ranges']['sodium_mean']['max'],
       value=140.0
   )
   anion_gap = st.number_input(
       'Anion Gap (mEq/L)',
       min_value=model_package['clinical_ranges']['aniongap_mean']['min'],
       max_value=model_package['clinical_ranges']['aniongap_mean']['max'],
       value=7.0
   )
with col2:
   glucose = st.number_input(
       'Glucose (mg/dL)',
       min_value=model_package['clinical_ranges']['glucose_mean']['min'],
       max_value=model_package['clinical_ranges']['glucose_mean']['max'],
       value=100.0
   )
   platelet = st.number_input(
       'Platelet Count (K/uL)',
       min_value=model_package['clinical_ranges']['platelet_mean']['min'],
       max_value=model_package['clinical_ranges']['platelet_mean']['max'],
       value=200.0
   )

# Vital Signs
st.subheader('Vital Signs')
col1, col2 = st.columns(2)
with col1:
   temp = st.number_input(
       'Temperature (°C)',
       min_value=model_package['clinical_ranges']['temperature_mean']['min'],
       max_value=model_package['clinical_ranges']['temperature_mean']['max'],
       value=37.0
   )
   sbp = st.number_input(
       'Systolic Blood Pressure (mmHg)',
       min_value=model_package['clinical_ranges']['sbp_mean']['min'],
       max_value=model_package['clinical_ranges']['sbp_mean']['max'],
       value=110
   )
with col2:
   resp_rate = st.number_input(
       'Respiratory Rate (breaths/min)',
       min_value=model_package['clinical_ranges']['resp_rate_mean']['min'],
       max_value=model_package['clinical_ranges']['resp_rate_mean']['max'],
       value=16
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

# Create prediction button
if st.button('Predict Mortality Risk'):
   # Create input data in correct order
   input_data = pd.DataFrame([[
       apsiii, sapsii, bicarbonate, multiple_lines, anion_gap,
       cancer, temp, platelet, sodium, glucose, diabetes,
       age, liver_disease, sbp, resp_rate
   ]], columns=model_package['feature_names'])
   
   # Make prediction
   prediction_prob = model_package['model'].predict_proba(input_data)[0][1]
   
   # Determine risk level
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
   
   # Display interpretation
   st.subheader('Risk Level Interpretation')
   if risk_level == 'Low':
       st.write('Mortality risk is relatively low. Consider standard monitoring protocols.')
   elif risk_level == 'Medium':
       st.write('Elevated mortality risk. Consider increased monitoring and preventive measures.')
   else:
       st.write('High mortality risk. Consider intensive monitoring and preventive interventions.')

# Add footer with model performance info
st.sidebar.markdown('### Model Information')
st.sidebar.write(f"AUC: {model_package['metrics']['auc']:.3f}")
st.sidebar.write(f"Brier Score: {model_package['metrics']['brier']:.3f}")

