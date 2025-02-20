import streamlit as st
import joblib
import os
import numpy as np
import urllib.request
import pandas as pd

# Page title and intro
st.set_page_config(page_title="30-Day Mortality Upon Central Line Insertion", layout="centered")
st.title("30-Day Mortality Upon Central Line Insertion")
st.markdown("**Note:** This tool is for **self-learning purposes only** and is not intended for clinical decision-making.")

# Model file settings
MODEL_FILENAME = "xgboost_mortality.pkl"
MODEL_URL = "https://raw.githubusercontent.com/acrana/Metis-Techo/main/MISC/Streamlit/Linemortality/xgboost_mortality.pkl"

# Download the model if missing
if not os.path.exists(MODEL_FILENAME):
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
    except Exception as e:
        st.error(f"Model download failed: {e}")
        st.stop()

# Load the model
try:
    model = joblib.load(MODEL_FILENAME)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define feature categories
binary_features = ['cancer', 'cva', 'rrt', 'liver_disease', 'multiple_lines']
numeric_features = ['apsiii_score', 'sapsii_score', 'age', 'inr_mean', 'aniongap_mean', 
                    'pt_mean', 'sodium_mean', 'resp_rate_mean', 'temperature_mean', 'spo2_mean']

st.subheader("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Clinical Scores & History")
    age = st.number_input("Age", min_value=18, max_value=100, value=65)
    apsiii_score = st.number_input("APS-III Score", min_value=0, max_value=100, value=30)
    sapsii_score = st.number_input("SAPS-II Score", min_value=0, max_value=100, value=30)

    st.markdown("### Medical History")
    cancer = st.radio("Cancer", ["No", "Yes"])
    cva = st.radio("Stroke (CVA)", ["No", "Yes"])
    rrt = st.radio("Renal Replacement Therapy", ["No", "Yes"])
    liver_disease = st.radio("Liver Disease", ["No", "Yes"])
    multiple_lines = st.radio("Multiple Central Lines", ["No", "Yes"])

with col2:
    st.markdown("### Lab & Vital Sign Data")
    inr_mean = st.number_input("INR Mean", min_value=0.0, max_value=10.0, value=1.1)
    aniongap_mean = st.number_input("Anion Gap Mean", min_value=0.0, max_value=50.0, value=12.0)
    pt_mean = st.number_input("Prothrombin Time (PT Mean)", min_value=0.0, max_value=50.0, value=14.0)
    sodium_mean = st.number_input("Sodium Mean", min_value=120.0, max_value=160.0, value=140.0)
    resp_rate_mean = st.number_input("Respiratory Rate Mean", min_value=5.0, max_value=50.0, value=16.0)
    temperature_mean = st.number_input("Temperature Mean (°C)", min_value=30.0, max_value=42.0, value=37.0)
    spo2_mean = st.number_input("SpO2 Mean (%)", min_value=50.0, max_value=100.0, value=98.0)

# Convert categorical inputs to binary (0/1)
binary_mapping = {"No": 0, "Yes": 1}
cancer = binary_mapping[cancer]
cva = binary_mapping[cva]
rrt = binary_mapping[rrt]
liver_disease = binary_mapping[liver_disease]
multiple_lines = binary_mapping[multiple_lines]

# Prepare input for prediction
input_data = np.array([[apsiii_score, sapsii_score, cancer, age, cva, rrt, inr_mean, liver_disease, 
                         multiple_lines, aniongap_mean, pt_mean, sodium_mean, resp_rate_mean, 
                         temperature_mean, spo2_mean]])

st.markdown("---")

# Predict and show feature importance
if st.button("Predict 30-Day Mortality Risk", use_container_width=True):
    prediction_prob = model.predict_proba(input_data)[0][1]
    
    # Get feature importances
    feature_importances = model.feature_importances_
    feature_names = ['apsiii_score', 'sapsii_score', 'cancer', 'age', 'cva', 'rrt', 'inr_mean', 
                     'liver_disease', 'multiple_lines', 'aniongap_mean', 'pt_mean', 'sodium_mean', 
                     'resp_rate_mean', 'temperature_mean', 'spo2_mean']
    
    # Multiply importances by input values to see what contributes most
    impact_scores = np.abs(feature_importances * input_data.flatten())
    
    # Sort features by impact
    sorted_indices = np.argsort(impact_scores)[::-1]
    top_factors = [(feature_names[i], impact_scores[i]) for i in sorted_indices[:5]]

    st.subheader("Prediction Result")
    st.write(f"Predicted **30-day mortality risk**: `{prediction_prob:.2%}`")

    if prediction_prob < 0.3:
        st.success("Low Risk")
    elif prediction_prob < 0.6:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

    # Show biggest contributing factors
    st.markdown("### Key Factors in Prediction")
    top_factors_df = pd.DataFrame(top_factors, columns=["Feature", "Impact Score"])
    st.dataframe(top_factors_df.style.format({"Impact Score": "{:.3f}"}))


