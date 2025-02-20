import streamlit as st
import joblib
import os
import numpy as np

# Title
st.title("🔍 30-Day Mortality Prediction App")
st.write("Enter patient information to predict 30-day mortality risk.")

# Load model
MODEL_FILENAME = "xgboost_mortality.pkl"

if not os.path.exists(MODEL_FILENAME):
    st.error("❌ Model file not found! Make sure `xgboost_mortality.pkl` is in the same directory.")
    st.stop()

try:
    model = joblib.load(MODEL_FILENAME)
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# Input fields
features = [
    'apsiii_score', 'sapsii_score', 'cancer', 'age', 'cva', 'rrt', 'inr_mean',
    'liver_disease', 'multiple_lines', 'aniongap_mean', 'pt_mean', 'sodium_mean',
    'resp_rate_mean', 'temperature_mean', 'spo2_mean'
]

st.sidebar.header("📝 Enter Patient Data")
user_input = {}

for feature in features:
    user_input[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()}:", 
                                                  min_value=0.0, max_value=1000.0, value=50.0)

input_data = np.array(list(user_input.values())).reshape(1, -1)

if st.sidebar.button("🔮 Predict Mortality Risk"):
    prediction_prob = model.predict_proba(input_data)[0][1]
    st.subheader("🛑 Prediction Result")
    st.write(f"Predicted **30-day mortality risk**: `{prediction_prob:.2%}`")

    if prediction_prob < 0.3:
        st.success("🟢 Low Risk")
    elif prediction_prob < 0.6:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

st.sidebar.subheader("🔍 Debugging: User Inputs")
st.sidebar.write(user_input)

