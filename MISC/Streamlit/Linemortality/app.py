import streamlit as st
import pandas as pd
import joblib
import os
from typing import Dict

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'mortality_prediction_model.joblib')

# Load model with error handling
@st.cache_resource
def load_model(model_path: str) -> Dict:
    """Load trained model package with validation"""
    try:
        model_package = joblib.load(model_path)
        required_keys = ['model', 'feature_names', 'clinical_ranges', 'metrics']
        if not all(key in model_package for key in required_keys):
            raise ValueError("Invalid model package structure")
        return model_package
    except Exception as e:
        st.error(f"""Failed to load model: {str(e)}
                 Please ensure the model file exists at {MODEL_PATH}""")
        st.stop()

model_package = load_model(MODEL_PATH)

# Helper functions
def get_default(feature: str) -> float:
    """Get clinically reasonable default value"""
    ranges = model_package['clinical_ranges'][feature]
    return round((ranges['min'] + ranges['max']) / 2, 1)

def validate_inputs(inputs: Dict) -> bool:
    """Check for clinically impossible combinations"""
    if inputs['sbp'] < 70:
        st.warning("Critically low blood pressure detected")
        return False
    if inputs['temperature'] < 32 or inputs['temperature'] > 42:
        st.error("Non-physiological temperature value")
        return False
    return True

# App layout
st.title('30-Day Mortality Prediction After Central Line Insertion')
st.markdown("""
**Clinical Decision Support Tool**  
Predicts mortality risk using physiological parameters and clinical characteristics.
""")

# Input sections
with st.expander("Patient Demographics", expanded=True):
    age = st.number_input(
        'Age (years)', 
        min_value=18.0,
        max_value=120.0,
        value=get_default('age'),
        step=1.0,
        help="Patient's chronological age"
    )

with st.expander("Clinical Scores", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        apsiii = st.number_input(
            'APSIII Score', 
            min_value=0.0,
            max_value=200.0,
            value=get_default('apsiii_score'),
            step=1.0,
            help="Acute Physiology Score III (0-200)"
        )
    with col2:
        sapsii = st.number_input(
            'SAPSII Score',
            min_value=0.0,
            max_value=200.0,
            value=get_default('sapsii_score'),
            step=1.0,
            help="Simplified Acute Physiology Score II (0-200)"
        )

# Laboratory Inputs
with st.expander("Laboratory Values"):
    lab_col1, lab_col2 = st.columns(2)
    with lab_col1:
        bicarbonate = st.number_input(
            'Bicarbonate (mEq/L)',
            min_value=5.0,
            max_value=50.0,
            value=get_default('bicarbonate_mean'),
            step=0.1,
            format="%.1f"
        )
        sodium = st.number_input(
            'Sodium (mEq/L)',
            min_value=120.0,
            max_value=160.0,
            value=get_default('sodium_mean'),
            step=1.0
        )
    with lab_col2:
        glucose = st.number_input(
            'Glucose (mg/dL)',
            min_value=50.0,
            max_value=1000.0,
            value=get_default('glucose_mean'),
            step=1.0
        )
        platelet = st.number_input(
            'Platelet Count (K/uL)',
            min_value=10.0,
            max_value=1000.0,
            value=get_default('platelet_mean'),
            step=10.0
        )

# Vital Signs
with st.expander("Vital Signs"):
    vital_col1, vital_col2 = st.columns(2)
    with vital_col1:
        temp = st.number_input(
            'Temperature (°C)',
            min_value=32.0,
            max_value=42.0,
            value=get_default('temperature_mean'),
            step=0.1,
            format="%.1f"
        )
    with vital_col2:
        sbp = st.number_input(
            'Systolic BP (mmHg)',
            min_value=50,
            max_value=250,
            value=int(get_default('sbp_mean')),
            step=1
        )
    resp_rate = st.number_input(
        'Respiratory Rate (/min)',
        min_value=5,
        max_value=50,
        value=int(get_default('resp_rate_mean')),
        step=1
    )

# Comorbidities
with st.expander("Comorbidities"):
    com_col1, com_col2 = st.columns(2)
    with com_col1:
        cancer = st.checkbox('Active Cancer')
        diabetes = st.checkbox('Diabetes Mellitus')
    with com_col2:
        liver_disease = st.checkbox('Chronic Liver Disease')
        multiple_lines = st.checkbox('Multiple Central Lines')

# Prediction Logic
if st.button('Calculate Mortality Risk'):
    input_data = {
        'apsiii_score': apsiii,
        'sapsii_score': sapsii,
        'bicarbonate_mean': bicarbonate,
        'multiple_lines': int(multiple_lines),
        'aniongap_mean': 12.0,  # Example fixed value from model
        'cancer': int(cancer),
        'temperature_mean': temp,
        'platelet_mean': platelet,
        'sodium_mean': sodium,
        'glucose_mean': glucose,
        'diabetes': int(diabetes),
        'age': age,
        'liver_disease': int(liver_disease),
        'sbp_mean': sbp,
        'resp_rate_mean': resp_rate
    }
    
    if not validate_inputs(input_data):
        st.error("Invalid input values detected")
        st.stop()
    
    try:
        df = pd.DataFrame([input_data])[model_package['feature_names']]
        proba = model_package['model'].predict_proba(df)[0][1]
        
        # Risk stratification
        risk_thresholds = model_package.get('risk_thresholds', [0.2, 0.4])
        risk_level = (
            'High' if proba >= risk_thresholds[1] else
            'Medium' if proba >= risk_thresholds[0] else 'Low'
        )
        
        # Display results
        st.subheader("Risk Assessment")
        st.metric(label="30-Day Mortality Probability", 
                value=f"{proba:.1%}", 
                help="Predicted probability of mortality within 30 days")
        
        if risk_level == 'High':
            st.error(f"High Risk: Consider ICU admission and aggressive monitoring")
        elif risk_level == 'Medium':
            st.warning(f"Moderate Risk: Recommend close monitoring and reassessment")
        else:
            st.success(f"Low Risk: Routine monitoring recommended")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# Model information sidebar
with st.sidebar:
    st.subheader("Model Details")
    st.markdown(f"""
    - **AUC**: {model_package['metrics']['auc']:.3f}  
    - **Brier Score**: {model_package['metrics']['brier']:.3f}  
    - **Features Used**: {len(model_package['feature_names'])}  
    - **Training Data**: MIMIC-IV v2.0  
    - **Updated**: 2023-09-01
    """)
    st.caption("""
    **Clinical Notes**:  
    This tool should support but not replace clinical judgment.  
    Always consider individual patient circumstances.
    """)
