import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("xgboost_mortality_final_swapped.pkl")

# Define the input fields for the top 15 features
st.title("30-Day Mortality Prediction App")
st.write("Enter patient information to predict 30-day mortality risk.")

# Create input fields for each feature
features = ['apsiii_score', 'sapsii_score', 'cancer', 'age', 'cva', 'rrt', 'inr_mean',
            'liver_disease', 'multiple_lines', 'aniongap_mean', 'pt_mean', 'sodium_mean',
            'resp_rate_mean', 'temperature_mean', 'spo2_mean']

input_values = []
for feature in features:
    value = st.number_input(f"Enter {feature}:", min_value=0.0, max_value=1000.0, value=50.0)
    input_values.append(value)

# Convert input values into a NumPy array and reshape it
input_array = np.array(input_values).reshape(1, -1)

# Predict the mortality risk when user clicks the button
if st.button("Predict Mortality Risk"):
    prediction_prob = model.predict_proba(input_array)[0][1]
    st.write(f"Predicted **30-day mortality risk**: {prediction_prob:.2%}")
