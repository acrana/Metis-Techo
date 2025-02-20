import os
import streamlit as st
import joblib
import urllib.request

# Streamlit App Title
st.title("🔍 30-Day Mortality Prediction App")
st.write("Enter patient information to predict 30-day mortality risk.")

# Define model filename and GitHub raw URL
MODEL_FILENAME = "xgboost_mortality.pkl"
MODEL_URL = "https://raw.githubusercontent.com/acrana/Metis-Techo/main/MISC/Streamlit/Linemortality/xgboost_mortality.pkl"  # ✅ Direct link to raw file

# Download the model if it's missing
if not os.path.exists(MODEL_FILENAME):
    st.write("⚠️ Model file not found! Downloading from GitHub...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        st.success("✅ Model downloaded successfully!")
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
import numpy as np
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
st.sidebar.write(user_input)
