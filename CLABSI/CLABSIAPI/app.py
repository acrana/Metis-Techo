iimport streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the model package from the same directory
model_package = joblib.load('mortality_prediction_model.joblib')
model = model_package['model']
feature_names = model_package['feature_names']
clinical_ranges = model_package['clinical_ranges']
metrics = model_package['metrics']

# Streamlit app configuration
st.set_page_config(page_title="Mortality Prediction", layout="wide")

# Title and description
st.title("30-Day Mortality Prediction After Central Line Insertion")
st.markdown("""
    This tool predicts the probability of 30-day mortality following central line insertion based on patient data.
    Enter the values below and click **Predict** to see the result.
""")

# Sidebar for model info
st.sidebar.header("Model Information")
st.sidebar.write("**Top 15 Features Used:**")
for i, feature in enumerate(feature_names, 1):
    st.sidebar.write(f"{i}. {feature.replace('_mean', '')}")
st.sidebar.write(f"**Test AUC:** {metrics['auc']:.3f}")
st.sidebar.write(f"**Test Brier Score:** {metrics['brier']:.3f}")

# Organize inputs into two columns
col1, col2 = st.columns(2)

# Input dictionary
input_data = {}

# Helper function to clean feature names
def clean_name(feature):
    return feature.replace('_mean', '').replace('_score', '').upper()

# Column 1: Demographics, Vitals, Binary Features
with col1:
    st.subheader("Demographics & Vitals")
    input_data['age'] = st.number_input("Age (years)", min_value=18, max_value=130, value=65, step=1)
    input_data['mbp_mean'] = st.number_input("Mean Blood Pressure (mmHg)", min_value=40.0, max_value=140.0, value=75.0, step=0.1)
    input_data['resp_rate_mean'] = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=50, value=20, step=1)
    input_data['temperature_mean'] = st.number_input("Temperature (°C)", min_value=32.0, max_value=42.0, value=37.0, step=0.1)
    
    st.subheader("Conditions")
    input_data['cancer'] = 1 if st.checkbox("Cancer Present", value=True) else 0
    input_data['multiple_lines'] = 1 if st.checkbox("Multiple Lines Inserted", value=False) else 0

# Column 2: Labs and Scores
with col2:
    st.subheader("Laboratory Values")
    input_data['wbc_mean'] = st.number_input("WBC (x10^9/L)", min_value=0.0, max_value=50.0, value=13.0, step=0.1)
    input_data['aniongap_mean'] = st.number_input("Anion Gap (mEq/L)", min_value=0.0, max_value=40.0, value=14.0, step=0.1)
    input_data['bicarbonate_mean'] = st.number_input("Bicarbonate (mEq/L)", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    input_data['creatinine_mean'] = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=1.5, step=0.1)
    input_data['chloride_mean'] = st.number_input("Chloride (mEq/L)", min_value=70.0, max_value=140.0, value=104.0, step=0.1)
    input_data['sodium_mean'] = st.number_input("Sodium (mEq/L)", min_value=110.0, max_value=160.0, value=138.0, step=0.1)
    
    st.subheader("Clinical Scores")
    input_data['sofa_score'] = st.slider("SOFA Score", min_value=0, max_value=24, value=7, step=1)
    input_data['apsiii_score'] = st.slider("APS III Score", min_value=0, max_value=200, value=50, step=1)
    input_data['sapsii_score'] = st.slider("SAPS II Score", min_value=0, max_value=150, value=40, step=1)

# Prediction button and result
if st.button("Predict Mortality Risk", key="predict_btn"):
    # Create DataFrame from inputs
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prob = model.predict_proba(input_df)[0, 1]
    
    # Display result with color-coded risk
    st.subheader("Prediction Result")
    if prob < 0.2:
        st.success(f"Predicted 30-day mortality probability: **{prob:.3f}** (Low Risk)")
    elif prob < 0.4:
        st.warning(f"Predicted 30-day mortality probability: **{prob:.3f}** (Medium Risk)")
    else:
        st.error(f"Predicted 30-day mortality probability: **{prob:.3f}** (High Risk)")

    # Feature importance plot
    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    xgb.plot_importance(model, max_num_features=10, ax=ax)
    plt.title("Top 10 Feature Importance")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.write("Developed using MIMIC-IV data. Model trained on 17,191 ICU stays with central lines.")
