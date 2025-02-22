import streamlit as st
import joblib
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "final_xgb_model.pkl")
try:
    model = joblib.load(model_path)
    model_feature_names = model.get_booster().feature_names
    explainer = shap.Explainer(model)
except FileNotFoundError:
    st.error("Model file 'final_xgb_model.pkl' not found in the app directory. Please ensure it’s uploaded to the GitHub repo.")
    st.stop()

st.set_page_config(page_title="CLABSI Risk Prediction", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #ffffff; margin-bottom: 5px; }
    .disclaimer { font-size: 14px; color: #bdc3c7; font-style: italic; margin-bottom: 20px; }
    .subheader { font-size: 20px; color: #34495e; margin-top: 10px; }
    .sidebar .sidebar-content { background-color: #ecf0f1; padding: 10px; }
    .stButton>button { background-color: #e74c3c; color: white; font-weight: bold; border-radius: 5px; }
    .stButton>button:hover { background-color: #c0392b; }
    .header-bg { background-color: #2c3e50; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .stNumberInput { width: 200px; }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="header-bg">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">CLABSI Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">For self learning and educational purposes</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("Predicts Central Line-Associated Bloodstream Infection (CLABSI) risk based on patient and line data.")

st.sidebar.header("Model Information")
st.sidebar.write("**Features Used:**")
for i, feature in enumerate(model_feature_names, 1):
    st.sidebar.write(f"{i}. {feature}")
st.sidebar.write("**Model Type:** XGBoost with SHAP Explanations")

default_values = {
    "admission_age": 50, "gender": 1, "has_diabetes": 0, "has_cancer": 0, "has_liver": 0,
    "has_chf": 0, "has_cva": 0, "days_since_last_dressing_change": 0, "chg_adherence_ratio": 0.5,
    "wbc_mean": 8.0, "plt_mean": 200.0, "creat_mean": 1.0, "inr_mean": 1.0, "pt_mean": 12.0,
    "sofa_score": 2, "apsiii": 50, "sapsii": 30
}

if 'clabsi_input_data' not in st.session_state:
    st.session_state.clabsi_input_data = default_values.copy()

input_data = st.session_state.clabsi_input_data

clinical_ranges = {
    "admission_age": {"min": 0, "max": 120}, "wbc_mean": {"min": 0.0, "max": 50.0},
    "plt_mean": {"min": 0.0, "max": 1000.0}, "creat_mean": {"min": 0.0, "max": 20.0},
    "inr_mean": {"min": 0.0, "max": 10.0}, "pt_mean": {"min": 0.0, "max": 50.0},
    "sofa_score": {"min": 0, "max": 24}, "apsiii": {"min": 0, "max": 200}, "sapsii": {"min": 0, "max": 150},
    "chg_adherence_ratio": {"min": 0.0, "max": 1.0}
}

def validate_inputs(inputs):
    for feature, value in inputs.items():
        if feature in clinical_ranges:
            min_val = clinical_ranges[feature]["min"]
            max_val = clinical_ranges[feature]["max"]
            if not (min_val <= value <= max_val):
                st.error(f"{feature.replace('_mean', '').upper()} must be between {min_val} and {max_val}.")
                return False
    return True

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    with st.expander("Patient Data", expanded=True):
        st.markdown('<div class="subheader">Patient Data</div>', unsafe_allow_html=True)
        input_data['admission_age'] = st.number_input("Line Age (days)", min_value=0, max_value=120, value=input_data['admission_age'], step=1, key="line_age")
        input_data['gender'] = st.selectbox("Gender", [0, 1], index=input_data['gender'], format_func=lambda x: "Female" if x == 0 else "Male", key="gender")

with col2:
    with st.expander("Conditions", expanded=True):
        st.markdown('<div class="subheader">Conditions</div>', unsafe_allow_html=True)
        input_data['has_diabetes'] = 1 if st.checkbox("Diabetes", value=bool(input_data['has_diabetes']), key="diabetes") else 0
        input_data['has_cancer'] = 1 if st.checkbox("Cancer", value=bool(input_data['has_cancer']), key="cancer") else 0
        input_data['has_liver'] = 1 if st.checkbox("Liver Disease", value=bool(input_data['has_liver']), key="liver") else 0
        input_data['has_chf'] = 1 if st.checkbox("CHF", value=bool(input_data['has_chf']), key="chf") else 0
        input_data['has_cva'] = 1 if st.checkbox("CVA", value=bool(input_data['has_cva']), key="cva") else 0

with col3:
    with st.expander("Line Care & Scores", expanded=True):
        st.markdown('<div class="subheader">Line Care & Scores</div>', unsafe_allow_html=True)
        input_data['chg_adherence_ratio'] = 1.0 - st.slider("CHG Adherence (0=None, 1=Perfect)", 0.0, 1.0, 1.0 - input_data['chg_adherence_ratio'], step=0.05, key="chg_adherence")
        input_data['sofa_score'] = st.number_input("SOFA Score", min_value=0, max_value=24, value=input_data['sofa_score'], step=1, key="sofa")
        input_data['apsiii'] = st.number_input("APSIII Score", min_value=0, max_value=200, value=input_data['apsiii'], step=1, key="apsiii")
        input_data['sapsii'] = st.number_input("SAPSII Score", min_value=0, max_value=150, value=input_data['sapsii'], step=1, key="sapsii")

with st.expander("Laboratory Values", expanded=True):
    st.markdown('<div class="subheader">Laboratory Values</div>', unsafe_allow_html=True)
    col_lab1, col_lab2, col_lab3, col_lab4, col_lab5 = st.columns(5)
    with col_lab1:
        input_data['wbc_mean'] = st.number_input("WBC (x10⁹/L)", min_value=0.0, max_value=50.0, value=input_data['wbc_mean'], step=0.1, key="wbc")
    with col_lab2:
        input_data['plt_mean'] = st.number_input("Platelets (K/uL)", min_value=0.0, max_value=1000.0, value=input_data['plt_mean'], step=1.0, key="plt")
    with col_lab3:
        input_data['creat_mean'] = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=input_data['creat_mean'], step=0.1, key="creat")
    with col_lab4:
        input_data['inr_mean'] = st.number_input("INR", min_value=0.0, max_value=10.0, value=input_data['inr_mean'], step=0.1, key="inr")
    with col_lab5:
        input_data['pt_mean'] = st.number_input("PT (seconds)", min_value=0.0, max_value=50.0, value=input_data['pt_mean'], step=0.1, key="pt")

st.markdown("---")
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    predict_clicked = st.button("Predict CLABSI Risk", key="predict_btn")
with col_btn2:
    if st.button("Reset Inputs", key="reset_btn"):
        st.session_state.clabsi_input_data = default_values.copy()
        st.rerun()

if predict_clicked:
    if validate_inputs(input_data):
        with st.spinner("Calculating CLABSI risk..."):
            df = pd.DataFrame([input_data])
            df_transformed = df.reindex(columns=model_feature_names, fill_value=0)
            prediction = model.predict(df_transformed)
            probability = model.predict_proba(df_transformed)[:, 1]
            risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
            shap_values = explainer(df_transformed)
            shap_contribs = shap_values[0].values
            factors = list(zip(df_transformed.columns, shap_contribs))
            factors_sorted = sorted(factors, key=lambda x: abs(x[1]), reverse=True)

        with st.expander("Prediction Result", expanded=True):
            st.subheader("Prediction Result")
            if risk_level == "High Risk":
                st.error(f"CLABSI Risk: **{risk_level}** (Probability: {probability[0]:.2%})")
            else:
                st.success(f"CLABSI Risk: **{risk_level}** (Probability: {probability[0]:.2%})")

        with st.expander("Key Risk Factors (SHAP)", expanded=True):
            st.subheader("Key Risk Factors")
            st.markdown("_Top contributors to this prediction (positive increases risk, negative decreases risk):_")
            for feature_name, shap_val in factors_sorted[:5]:
                direction = "↑" if shap_val > 0 else "↓"
                st.write(f"**{feature_name}**: {direction} ({shap_val:.3f})")

        with st.expander("SHAP Waterfall Plot", expanded=True):
            st.subheader("SHAP Waterfall Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig)

st.markdown("---")
st.write("Developed as a personal project using clinical data for educational purposes.")
