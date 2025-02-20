import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import urllib.request

# Title of the Streamlit app
st.title("🔍 30-Day Mortality Prediction App")
st.write("Enter patient information to predict the 30-day mortality risk.")

# Path to the model file
MODEL_FILENAME = "xgboost_mortality.pkl"
MODEL_URL = "https://your-storage-link/xgboost_mortality.pkl"  # 🔄 Replace with actual storage URL if needed

# Check if the model exists, otherwise download it
if not os.path.exists(MODEL_FILENAME):
    st.write("⚠️ Model file not found! Trying to download...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        st.write("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"🚨 Failed to download model: {e}")
        st.stop()

# Load the model
try:
    model = joblib.load(MODEL_FILENAME)
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# List of required input features
features = [
    'apsiii_score', 'sapsii_score', 'cancer', 'age', 'cva', 'rrt', 'inr_mean',
    'liver_disease', 'multiple_lines', 'aniongap_mean', 'pt_mean', 'sodium_mean',
    'resp_rate_mean', 'temperature_mean', 'spo2_mean'
]

# Create input fields for user input
st.sidebar.header("📝 Enter Patient Data")
user_input = {}

for feature in features:
    user_input[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()}:", 
                                                  min_value=0.0, max_value=1000.0, value=50.0)

# Convert user input to DataFrame for prediction
input_data = np.array(list(user_input.values())).reshape(1, -1)

# Predict button
if st.sidebar.button("🔮 Predict Mortality Risk"):
    prediction_prob = model.predict_proba(input_data)[0][1]
    st.subheader("🛑 Prediction Result")
    st.write(f"Predicted **30-day mortality risk**: `{prediction_prob:.2%}`")

    # Risk level categorization
    if prediction_prob < 0.3:
        st.success("🟢 Low Risk")
    elif prediction_prob < 0.6:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

# Debugging: Show user inputs (for dev testing)
st.sidebar.subheader("🔍 Debugging: User Inputs")
st.sidebar.write(pd.DataFrame(user_input, index=[0]))
