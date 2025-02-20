import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model package
model_filename = 'mortality_prediction_model.joblib'
model_path = os.path.join(os.path.dirname(__file__), model_filename)

if not os.path.exists(model_path):
    st.error(f"⚠️ Model file '{model_filename}' not found! Ensure it's in the same directory as this script.")
    st.stop()

model_package = joblib.load(model_path)
model = model_package['model']
feature_names = model_package['feature_names']
feature_ranges = model_package['feature_ranges']

# Load the scaler if it was used during training
scaler = model_package.get('scaler', None)

if not scaler:
    st.error("Scaler not found in model package. Ensure it's included during training.")
    st.stop()

# UI Title
st.title('30-Day Mortality Prediction After Central Line Insertion')

st.write("""
### Enter Patient Information
Provide the required patient parameters to estimate mortality risk.
""")

# Arrange inputs in two columns
col1, col2 = st.columns(2)
input_data = {}

# Create input fields using a form
with st.form("input_form"):
    for i, feature in enumerate(feature_names):
        with col1 if i % 2 == 0 else col2:  # Distribute inputs across two columns
            if feature in ['cancer', 'liver_disease', 'cva', 'rrt', 'multiple_lines']:
                input_data[feature] = st.checkbox(f'{feature.replace("_", " ").title()}')
            
            else:
                # Accept real-world values instead of scaled values
                input_data[feature] = st.number_input(
                    f'{feature.replace("_", " ").title()}',
                    min_value=float(feature_ranges[feature]['min']),
                    max_value=float(feature_ranges[feature]['max']),
                    value=float(feature_ranges[feature]['mean']),
                    step=0.01
                )

    # Submit button inside the form
    submit_button = st.form_submit_button("Predict Risk")

if submit_button:
    # Convert input into DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply MinMax Scaling to match model training data
    scaled_input = scaler.transform(input_df)

    # Make prediction
    probability = model.predict_proba(scaled_input)[0][1]
    
    # Define risk category
    if probability < 0.2:
        risk_category, color = 'Low', 'green'
    elif probability < 0.4:
        risk_category, color = 'Medium', 'orange'
    else:
        risk_category, color = 'High', 'red'
    
    # Display results
    st.subheader("Prediction Results")
    st.metric(label="30-Day Mortality Probability", value=f"{probability:.1%}")
    st.markdown(f"**Risk Category:** <span style='color:{color}; font-size:22px;'>{risk_category}</span>", unsafe_allow_html=True)

    # Display risk interpretation
    st.subheader("Risk Interpretation")
    if risk_category == 'Low':
        st.write("- Standard monitoring recommended")
        st.write("- Review risk factors for potential modification")
    elif risk_category == 'Medium':
        st.write("- Enhanced monitoring recommended")
        st.write("- Consider early intervention for modifiable risk factors")
        st.write("- Regular reassessment of clinical status")
    else:
        st.write("- Close monitoring required")
        st.write("- Immediate attention to modifiable risk factors")
        st.write("- Consider intensive care management")
        st.write("- Early goals of care discussion recommended")

# Sidebar Information
st.sidebar.subheader('Model Information')
st.sidebar.write(f"Model Performance (AUC): **{model_package['metrics']['auc']:.3f}**")
st.sidebar.write("**Risk Categories:**")
st.sidebar.write("- Low: < 20% mortality risk")
st.sidebar.write("- Medium: 20-40% mortality risk")
st.sidebar.write("- High: > 40% mortality risk")

