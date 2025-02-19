import streamlit as st
import numpy as np
import os
import joblib

st.sidebar.markdown("# Disclaimer\n**Please note:** This app is a personal project and is not intended for serious medical use. It is designed for educational and demonstration purposes only.")

model_path = os.path.join(os.path.dirname(__file__), "mortality_model.pkl")
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

expected_features = [
    "has_tracheostomy", 
    "n_line_types_1", "n_line_types_2", "n_line_types_3", "n_line_types_4",
    "apsiii_score", "cancer", "cva", "rrt", "sapsii_score",
    "inr_mean", "age", "has_ostomy", "liver_disease", "aniongap_mean"
]

def map_lines(num):
    if num == 1: return [1, 0, 0, 0]
    elif num == 2: return [0, 1, 0, 0]
    elif num == 3: return [0, 0, 1, 0]
    else: return [0, 0, 0, 1]

def predict_mortality(input_dict):
    data = np.array([input_dict[feat] for feat in expected_features], dtype=float).reshape(1, -1)
    data = np.nan_to_num(data)
    return model.predict_proba(data)[0, 1]

st.title("30-Day Mortality Prediction App")
st.markdown("Enter patient information below:")

input_data = {}

st.subheader("Central Line Information")
num_lines = st.number_input("Number of Central Lines", min_value=1, max_value=10, value=1, step=1, help="Enter how many different central line types the patient has.")
lines = map_lines(num_lines)
input_data["n_line_types_1"] = lines[0]
input_data["n_line_types_2"] = lines[1]
input_data["n_line_types_3"] = lines[2]
input_data["n_line_types_4"] = lines[3]

st.subheader("Patient Characteristics")
col1, col2 = st.columns(2)
with col1:
    has_tracheostomy = st.checkbox("Has Tracheostomy", value=False)
    cancer = st.checkbox("Cancer", value=False)
    cva = st.checkbox("CVA", value=False)
with col2:
    dialysis = st.checkbox("Dialysis", value=False, help="Dialysis (model feature: rrt)")
    has_ostomy = st.checkbox("Has Ostomy", value=False)
    liver_disease = st.checkbox("Liver Disease", value=False)
input_data["has_tracheostomy"] = 1 if has_tracheostomy else 0
input_data["cancer"] = 1 if cancer else 0
input_data["cva"] = 1 if cva else 0
input_data["rrt"] = 1 if dialysis else 0
input_data["has_ostomy"] = 1 if has_ostomy else 0
input_data["liver_disease"] = 1 if liver_disease else 0

st.subheader("Clinical Measurements")
col3, col4 = st.columns(2)
with col3:
    apsiii_score = st.number_input("APSIII Score", value=50.0)
    sapsii_score = st.number_input("SAPSII Score", value=30.0)
    inr_mean = st.number_input("INR Mean", value=1.0)
with col4:
    age = st.number_input("Age", value=65.0)
    aniongap_mean = st.number_input("Anion Gap Mean", value=14.0)
input_data["apsiii_score"] = apsiii_score
input_data["sapsii_score"] = sapsii_score
input_data["inr_mean"] = inr_mean
input_data["age"] = age
input_data["aniongap_mean"] = aniongap_mean

if st.button("Predict 30-Day Mortality Risk"):
    risk = predict_mortality(input_data)
    st.subheader(f"Predicted 30-Day Mortality Risk: {risk:.2%}")
    if risk >= 0.26:
        st.error("High risk of mortality within 30 days.")
    else:
        st.success("Low risk of mortality within 30 days.")
