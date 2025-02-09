import streamlit as st
import joblib
import pandas as pd
import os
import time

# --------------------------------------------------------------------------------
# 1. File Paths for Model
# --------------------------------------------------------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "final_xgb_model.pkl")

# Load the XGBoost model
model = joblib.load(model_path)
model_feature_names = model.get_booster().feature_names  # columns the model expects

# --------------------------------------------------------------------------------
# 2. App Title & Disclaimer
# --------------------------------------------------------------------------------
st.title("Line Risk Prediction")
st.markdown(
    "**Please note: This app is a personal project and is not intended for "
    "serious medical use. It is designed for educational and demonstration "
    "purposes only.**"
)

# --------------------------------------------------------------------------------
# 3. UI Layout
# --------------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    line_age = st.number_input("Line Age", min_value=0, max_value=120, value=50)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    has_diabetes = st.checkbox("Has Diabetes")
    has_cancer = st.checkbox("Has Cancer")
    has_liver = st.checkbox("Has Liver Disease")
    has_chf = st.checkbox("Has CHF")
    has_cva = st.checkbox("Has CVA")

    # User sees 0 = worst adherence, 1 = best adherence
    user_chg_adherence = st.slider(
        "CHG Adherence Ratio (0 = No Adherence, 1 = Perfect Adherence)",
        0.0, 1.0, 0.5
    )
    # Invert for the model if it learned the opposite relationship
    model_chg_adherence = 1.0 - user_chg_adherence

with col2:
    wbc_mean = st.number_input("WBC Mean", min_value=0.0, value=8.0)
    plt_mean = st.number_input("Platelet Mean", min_value=0.0, value=200.0)
    creat_mean = st.number_input("Creatinine Mean", min_value=0.0, value=1.0)
    inr_mean = st.number_input("INR Mean", min_value=0.0, value=1.0)
    pt_mean = st.number_input("PT Mean", min_value=0.0, value=12.0)
    sofa_score = st.number_input("SOFA Score", min_value=0, value=2)
    apsiii = st.number_input("APSIII Score", min_value=0, value=50)
    sapsii = st.number_input("SAPSII Score", min_value=0, value=30)

# --------------------------------------------------------------------------------
# 4. Prediction Logic (With a Spinner)
# --------------------------------------------------------------------------------
if st.button("Predict"):
    with st.spinner("Calculating your risk..."):
        # Prepare the input dictionary
        input_data = {
            "admission_age": line_age,
            "gender": int(gender),
            "has_diabetes": int(has_diabetes),
            "has_cancer": int(has_cancer),
            "has_liver": int(has_liver),
            "has_chf": int(has_chf),
            "has_cva": int(has_cva),
            # Hidden from UI, required by the model, set to default 0
            "days_since_last_dressing_change": 0,
            # Pass the inverted slider value if the model was trained inversely
            "chg_adherence_ratio": model_chg_adherence,
            "wbc_mean": wbc_mean,
            "plt_mean": plt_mean,
            "creat_mean": creat_mean,
            "inr_mean": inr_mean,
            "pt_mean": pt_mean,
            "sofa_score": sofa_score,
            "apsiii": apsiii,
            "sapsii": sapsii
        }

        # Build DataFrame for prediction
        df = pd.DataFrame([input_data])
        df_transformed = df.reindex(columns=model_feature_names, fill_value=0)

        # Model prediction
        prediction = model.predict(df_transformed)
        probability = model.predict_proba(df_transformed)[:, 1]

        time.sleep(1)  # Optional small pause for demonstration

    st.success("Done!")
    # Interpret results
    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.subheader(f"Prediction: {risk_level}")
    st.subheader(f"Probability: {probability[0]:.2%}")

