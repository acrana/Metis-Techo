import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import xgboost as xgb

# Load the model from the same directory
model_path = os.path.join(os.path.dirname(__file__), 'mortality_prediction_model.joblib')
try:
    model_package = joblib.load(model_path)
    model = model_package['model']
    feature_names = model_package['feature_names']
    clinical_ranges = model_package['clinical_ranges']
    metrics = model_package['metrics']
except FileNotFoundError:
    st.error("Model file 'mortality_prediction_model.joblib' not found in the app directory. Please ensure it’s uploaded to the GitHub repo.")
    st.stop()

# Streamlit app configuration
st.set_page_config(page_title="Mortality Prediction Tool", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #ffffff; margin-bottom: 5px; }
    .disclaimer { font-size: 14px; color: #bdc3c7; font-style: italic; margin-bottom: 20px; }
    .subheader { font-size: 20px; color: #34495e; margin-top: 10px; }
    .sidebar .sidebar-content { background-color: #ecf0f1; padding: 10px; }
    .stButton>button { background-color: #3498db; color: white; font-weight: bold; border-radius: 5px; }
    .stButton>button:hover { background-color: #2980b9; }
    .header-bg { background-color: #2c3e50; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# Header with title and disclaimer
with st.container():
    st.markdown('<div class="header-bg">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">30-Day Mortality Prediction After Central Line Insertion</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">For self learning and educational purposes</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
    This clinical decision support tool predicts the probability of 30-day mortality following central line insertion based on patient data from MIMIC-IV.
    Enter values below and click **Predict** to assess risk.
""")

# Sidebar with model info
st.sidebar.header("Model Information")
st.sidebar.write("**Features Used (15):**")
for i, feature in enumerate(feature_names, 1):
    st.sidebar.write(f"{i}. {feature.replace('_mean', '').replace('_score', '')}")
st.sidebar.write(f"**Test AUC:** {metrics['auc']:.3f}")
st.sidebar.write(f"**Test Brier Score:** {metrics['brier']:.3f}")

# Input dictionary with default values
default_values = {
    'age': 65, 'mbp_mean': 75.0, 'resp_rate_mean': 20.0, 'temperature_mean': 37.0,
    'cancer': 0, 'multiple_lines': 0, 'wbc_mean': 13.0, 'aniongap_mean': 14.0,
    'bicarbonate_mean': 22.0, 'creatinine_mean': 1.5, 'chloride_mean': 104.0,
    'sodium_mean': 138.0, 'sofa_score': 7.0, 'apsiii_score': 50.0, 'sapsii_score': 40.0
}

# Initialize session state for inputs
if 'input_data' not in st.session_state:
    st.session_state.input_data = default_values.copy()

input_data = st.session_state.input_data

# Helper function to clean feature names
def clean_name(feature):
    return feature.replace('_mean', '').replace('_score', '').upper()

# Validation function
def validate_inputs(inputs):
    for feature, value in inputs.items():
        if feature in clinical_ranges:
            min_val = clinical_ranges[feature]['min']
            max_val = clinical_ranges[feature]['max']
            if not (min_val <= value <= max_val):
                st.error(f"{clean_name(feature)} must be between {min_val} and {max_val}.")
                return False
    return True

# Two-column layout for inputs
col1, col2 = st.columns([1, 1])

# Column 1: Demographics, Vitals, Conditions
with col1:
    with st.expander("Demographics & Vitals", expanded=True):
        st.markdown('<div class="subheader">Demographics & Vitals</div>', unsafe_allow_html=True)
        input_data['age'] = st.number_input("Age (years)", min_value=18, max_value=130, value=input_data['age'], step=1, key="age")
        input_data['mbp_mean'] = st.number_input("Mean Blood Pressure (mmHg)", min_value=40.0, max_value=140.0, value=input_data['mbp_mean'], step=0.1, key="mbp")
        input_data['resp_rate_mean'] = st.number_input("Respiratory Rate (breaths/min)", min_value=5.0, max_value=50.0, value=input_data['resp_rate_mean'], step=0.1, key="resp_rate")
        input_data['temperature_mean'] = st.number_input("Temperature (°C)", min_value=32.0, max_value=42.0, value=input_data['temperature_mean'], step=0.1, key="temp")

    with st.expander("Conditions", expanded=True):
        st.markdown('<div class="subheader">Conditions</div>', unsafe_allow_html=True)
        input_data['cancer'] = 1 if st.checkbox("Cancer Present", value=bool(input_data['cancer']), key="cancer") else 0
        input_data['multiple_lines'] = 1 if st.checkbox("Multiple Lines Inserted", value=bool(input_data['multiple_lines']), key="multi_lines") else 0

# Column 2: Labs and Scores
with col2:
    with st.expander("Laboratory Values", expanded=True):
        st.markdown('<div class="subheader">Laboratory Values</div>', unsafe_allow_html=True)
        input_data['wbc_mean'] = st.number_input("WBC (x10⁹/L)", min_value=0.0, max_value=50.0, value=input_data['wbc_mean'], step=0.1, key="wbc")
        input_data['aniongap_mean'] = st.number_input("Anion Gap (mEq/L)", min_value=0.0, max_value=40.0, value=input_data['aniongap_mean'], step=0.1, key="aniongap")
        input_data['bicarbonate_mean'] = st.number_input("Bicarbonate (mEq/L)", min_value=10.0, max_value=50.0, value=input_data['bicarbonate_mean'], step=0.1, key="bicarb")
        input_data['creatinine_mean'] = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=input_data['creatinine_mean'], step=0.1, key="creat")
        input_data['chloride_mean'] = st.number_input("Chloride (mEq/L)", min_value=70.0, max_value=140.0, value=input_data['chloride_mean'], step=0.1, key="chloride")
        input_data['sodium_mean'] = st.number_input("Sodium (mEq/L)", min_value=110.0, max_value=160.0, value=input_data['sodium_mean'], step=0.1, key="sodium")

    with st.expander("Clinical Scores", expanded=True):
        st.markdown('<div class="subheader">Clinical Scores</div>', unsafe_allow_html=True)
        input_data['sofa_score'] = st.number_input("SOFA Score", min_value=0.0, max_value=24.0, value=input_data['sofa_score'], step=0.1, key="sofa")
        input_data['apsiii_score'] = st.number_input("APSIII Score", min_value=0.0, max_value=200.0, value=input_data['apsiii_score'], step=0.1, key="apsiii")
        input_data['sapsii_score'] = st.number_input("SAPSII Score", min_value=0.0, max_value=150.0, value=input_data['sapsii_score'], step=0.1, key="sapsii")

# Buttons
st.markdown("---")
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    predict_clicked = st.button("Predict Mortality Risk", key="predict_btn")
with col_btn2:
    if st.button("Reset Inputs", key="reset_btn"):
        st.session_state.input_data = default_values.copy()
        st.rerun()

# Prediction and results
if predict_clicked:
    if validate_inputs(input_data):
        with st.spinner("Calculating prediction..."):
            input_df = pd.DataFrame([input_data], columns=feature_names)
            prob = model.predict_proba(input_df)[0, 1]
        
        # Prediction result
        with st.expander("Prediction Result", expanded=True):
            st.subheader("Prediction Result")
            if prob < 0.2:
                st.success(f"Predicted 30-day mortality probability: **{prob:.3f}** (Low Risk)")
            elif prob < 0.4:
                st.warning(f"Predicted 30-day mortality probability: **{prob:.3f}** (Medium Risk)")
            else:
                st.error(f"Predicted 30-day mortality probability: **{prob:.3f}** (High Risk)")
            st.write(f"Model AUC: {metrics['auc']:.3f}, Brier Score: {metrics['brier']:.3f}")

        # Feature importance plot
        with st.expander("Feature Importance", expanded=True):
            st.subheader("Feature Importance")
            fig, ax = plt.subplots(figsize=(8, 6))
            xgb.plot_importance(model, max_num_features=10, ax=ax)
            plt.title("Top 10 Feature Importance", fontsize=14)
            st.pyplot(fig)

# Footer
st.markdown("---")
st.write("Developed using MIMIC-IV data. Model trained on 17,191 ICU stays with central lines.")
