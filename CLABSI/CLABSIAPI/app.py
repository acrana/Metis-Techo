import streamlit as st
import joblib
import pandas as pd
import os

# --------------------------------------------------------------------------------
# Locate directory of this file and build absolute paths for the model files
# --------------------------------------------------------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, "final_xgb_model.pkl")

# Load the model
model = joblib.load(model_path)

# If you prefer to load the feature list from the model itself:
model_feature_names = model.get_booster().feature_names

# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------
st.title("Line Risk Prediction (Hidden days_since_last_dressing_change)")

col1, col2 = st.columns(2)

with col1:
    line_age = st.number_input("Line Age", min_value=0, max_value=120)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    has_diabetes = st.checkbox("Has Diabetes")
    has_cancer = st.checkbox("Has Cancer")
    has_liver = st.checkbox("Has Liver Disease")
    has_chf = st.checkbox("Has CHF")
    has_cva = st.checkbox("Has CVA")
    chg_adherence_ratio = st.slider("CHG Adherence Ratio", 0.0, 1.0, 0.5)

with col2:
    wbc_mean = st.number_input("WBC Mean", min_value=0.0)
    plt_mean = st.number_input("Platelet Mean", min_value=0.0)
    creat_mean = st.number_input("Creatinine Mean", min_value=0.0)
    inr_mean = st.number_input("INR Mean", min_value=0.0)
    pt_mean = st.number_input("PT Mean", min_value=0.0)
    sofa_score = st.number_input("SOFA Score", min_value=0)
    apsiii = st.number_input("APSIII Score", min_value=0)
    sapsii = st.number_input("SAPSII Score", min_value=0)

# --------------------------------------------------------------------------------
# Predict button
# --------------------------------------------------------------------------------
if st.button("Predict"):
    # Create the input_data dictionary and set days_since_last_dressing_change as default=0
    input_data = {
        "admission_age": line_age,
        "gender": int(gender),
        "has_diabetes": int(has_diabetes),
        "has_cancer": int(has_cancer),
        "has_liver": int(has_liver),
        "has_chf": int(has_chf),
        "has_cva": int(has_cva),
        "days_since_last_dressing_change": 0,  # Hidden/default value
        "chg_adherence_ratio": chg_adherence_ratio,
        "wbc_mean": wbc_mean,
        "plt_mean": plt_mean,
        "creat_mean": creat_mean,
        "inr_mean": inr_mean,
        "pt_mean": pt_mean,
        "sofa_score": sofa_score,
        "apsiii": apsiii,
        "sapsii": sapsii
    }

    # Create DataFrame
    df = pd.DataFrame([input_data])

    # Reindex to match model features exactly (if needed)
    # Using the model's own feature names ensures perfect alignment
    df_transformed = df.reindex(columns=model_feature_names, fill_value=0)

    # Prediction
    prediction = model.predict(df_transformed)
    probability = model.predict_proba(df_transformed)[:, 1]

    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"

    st.header(f"Prediction: {risk_level}")
    st.subheader(f"Probability: {probability[0]:.2%}")
