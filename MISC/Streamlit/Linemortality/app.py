import streamlit as st
import joblib
import pandas as pd

# Load the model package
model_package = joblib.load('mortality_prediction_model.joblib')
model = model_package['model']
feature_names = model_package['feature_names']
clinical_ranges = model_package['clinical_ranges']
metrics = model_package['metrics']

# Streamlit app
st.title("30-Day Mortality Prediction After Central Line Insertion")
st.write("Enter patient data to predict 30-day mortality risk.")

# Input sliders/selectboxes for all 15 features
input_data = {}
for feature in feature_names:
    if feature in ['cancer', 'multiple_lines']:
        input_data[feature] = st.selectbox(f"{feature.replace('_mean', '')} (0 = No, 1 = Yes)", 
                                          options=[0, 1], key=feature)
    else:
        input_data[feature] = st.slider(f"{feature.replace('_mean', '')}", 
                                       min_value=float(clinical_ranges[feature]['min']), 
                                       max_value=float(clinical_ranges[feature]['max']), 
                                       value=float(clinical_ranges[feature]['min']) + 
                                               (clinical_ranges[feature]['max'] - clinical_ranges[feature]['min']) / 2, 
                                       key=feature)

# Predict when button is clicked
if st.button("Predict"):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prob = model.predict_proba(input_df)[0, 1]
    st.write(f"Predicted 30-day mortality probability: **{prob:.3f}**")
    st.write(f"Model AUC: {metrics['auc']:.3f}, Brier Score: {metrics['brier']:.3f}")

# Model info
st.sidebar.header("Model Info")
st.sidebar.write("Top 15 Features Used:")
for i, feature in enumerate(feature_names, 1):
    st.sidebar.write(f"{i}. {feature.replace('_mean', '')}")
st.sidebar.write(f"Test AUC: {metrics['auc']:.3f}")
st.sidebar.write(f"Test Brier Score: {metrics['brier']:.3f}")