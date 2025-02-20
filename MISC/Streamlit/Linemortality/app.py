import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Get the absolute path of the model file
model_filename = 'mortality_prediction_model.joblib'
model_path = os.path.join(os.path.dirname(__file__), model_filename)

# Check if the file exists
if not os.path.exists(model_path):
    st.error(f"⚠️ Model file '{model_filename}' not found! Ensure it's in the same directory as this script.")
    st.stop()

# Load the model package
model_package = joblib.load('mortality_prediction_model.joblib')
model = model_package['model']
feature_names = model_package['feature_names']
feature_ranges = model_package['feature_ranges']

st.title('30-Day Mortality Prediction After Central Line Insertion')

st.write("""
### Enter Patient Information
Please input the following values to predict 30-day mortality risk.
""")

# Create input fields for each feature
input_data = {}

for feature in feature_names:
    # Create appropriate input widgets based on feature type
    if feature in ['cancer', 'liver_disease', 'cva', 'rrt', 'multiple_lines']:
        # Binary features
        input_data[feature] = st.checkbox(f'{feature.replace("_", " ").title()}')
    
    elif feature == 'apsiii_score':
        input_data[feature] = st.slider(
            'APSIII Score (scaled)', 
            min_value=0.0,
            max_value=1.0,
            value=feature_ranges[feature]['mean'],
            step=0.01
        )
    
    elif feature == 'sapsii_score':
        input_data[feature] = st.slider(
            'SAPSII Score (scaled)', 
            min_value=0.0,
            max_value=1.0,
            value=feature_ranges[feature]['mean'],
            step=0.01
        )
    
    else:
        # Continuous features
        input_data[feature] = st.number_input(
            f'{feature.replace("_", " ").title()}',
            min_value=float(feature_ranges[feature]['min']),
            max_value=float(feature_ranges[feature]['max']),
            value=float(feature_ranges[feature]['mean']),
            step=0.01
        )

if st.button('Predict Risk'):
    # Create DataFrame with input data
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    probability = model.predict_proba(input_df)[0][1]
    
    # Determine risk category
    if probability < 0.2:
        risk_category = 'Low'
        color = 'green'
    elif probability < 0.4:
        risk_category = 'Medium'
        color = 'orange'
    else:
        risk_category = 'High'
        color = 'red'
    
    # Display results
    st.write('### Results')
    st.write(f'Predicted 30-day mortality probability: {probability:.1%}')
    st.markdown(f'Risk Category: <span style="color:{color}">{risk_category}</span>', unsafe_allow_html=True)
    
    # Display risk interpretation
    st.write('### Risk Interpretation')
    if risk_category == 'Low':
        st.write('- Standard monitoring recommended')
        st.write('- Review risk factors for potential modification')
    elif risk_category == 'Medium':
        st.write('- Enhanced monitoring recommended')
        st.write('- Consider early intervention for modifiable risk factors')
        st.write('- Regular reassessment of clinical status')
    else:
        st.write('- Close monitoring required')
        st.write('- Immediate attention to modifiable risk factors')
        st.write('- Consider intensive care management')
        st.write('- Early goals of care discussion recommended')

st.sidebar.write('### Model Information')
st.sidebar.write(f'Model Performance (AUC): {model_package["metrics"]["auc"]:.3f}')
st.sidebar.write('Risk Categories:')
st.sidebar.write('- Low: < 20% mortality risk')
st.sidebar.write('- Medium: 20-40% mortality risk')
st.sidebar.write('- High: > 40% mortality risk')
