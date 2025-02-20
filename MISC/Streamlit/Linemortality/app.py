import streamlit as st
import joblib
import os
import numpy as np
import urllib.request

# App Title
st.title("30-Day Mortality Prediction App")

# Disclaimer
st.write("**Note:** This tool is for self-learning purposes only and is not intended for clinical decision-making.")

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

# Load model
try:
    model = joblib.load(MODEL_FILENAME)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define feature categories
binary_features = ['cancer', 'cva', 'rrt', 'liver_disease', 'multiple_lines']
numeric_features = ['apsiii_score', 'sapsii_score', 'age', 'inr_mean', 'aniongap_mean', 
                    'pt_mean', 'sodium_mean', 'resp_rate_mean', 'temperature_mean', 'spo2_mean']

st.sidebar.header("Enter Patient Data")
user_input = {}

# Binary inputs (0 = No, 1 = Yes)
for feature in binary_features:
    user_input[feature] = st.sidebar.radio(f"{feature.replace('_', ' ').title()}:", [0, 1], index=0)

# Numeric inputs
for feature in numeric_features:
    default_values = {'age': 65, 'apsiii_score': 30, 'sapsii_score': 30}
    user_input[feature] = st.sidebar.number_input(
        f"{feature.replace('_', ' ').title()}:", 
        min_value=0.0, max_value=1000.0, 
        value=default_values.get(feature, 10)
    )

# Convert input to array
input_data = np.array(list(user_input.values())).reshape(1, -1)

# Predict
if st.sidebar.button("Predict Mortality Risk"):
    prediction_prob = model.predict_proba(input_data)[0][1]
    st.subheader("Prediction Result")
    st.write(f"Predicted 30-day mortality risk: `{prediction_prob:.2%}`")

    if prediction_prob < 0.3:
        st.success("Low Risk")
    elif prediction_prob < 0.6:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

st.sidebar.subheader("User Inputs")
st.sidebar.write(user_input)
