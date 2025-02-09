import streamlit as st
import joblib
import pandas as pd
import os

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "final_xgb_model.pkl")

model = joblib.load(model_path)
model_feature_names = model.get_booster().feature_names

st.title("Line Risk Prediction")

line_age = st.number_input("Line Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
has_diabetes = st.checkbox("Has Diabetes")
has_cancer = st.checkbox("Has Cancer")
has_liver = st.checkbox("Has Liver Disease")
has_chf = st.checkbox("Has CHF")
has_cva = st.checkbox("Has CVA")

# -------------------------------------------------------------------
# Invert the slider value for CHG ratio
# The user sees 0.0 = worst adherence -> 1.0 = best adherence
# But the model internally receives the opposite.
# -------------------------------------------------------------------
user_chg_adherence = st.slider(
    "CHG Adherence Ratio (0.0 = no adherence, 1.0 = full adherence)",
    0.0, 1.0, 0.5
)
# If the model learned the inverse relationship,
# we pass the flipped value to the model:
model_chg_adherence = 1.0 - user_chg_adherence

wbc_mean = st.number_input("WBC Mean", min_value=0.0)
plt_mean = st.number_input("Platelet Mean", min_value=0.0)
creat_mean = st.number_input("Creatinine Mean", min_value=0.0)
inr_mean = st.number_input("INR Mean", min_value=0.0)
pt_mean = st.number_input("PT Mean", min_value=0.0)
sofa_score = st.number_input("SOFA Score", min_value=0)
apsiii = st.number_input("APSIII Score", min_value=0)
sapsii = st.number_input("SAPSII Score", min_value=0)

if st.button("Predict"):
    input_data = {
        "admission_age": line_age,
        "gender": int(gender),
        "has_diabetes": int(has_diabetes),
        "has_cancer": int(has_cancer),
        "has_liver": int(has_liver),
        "has_chf": int(has_chf),
        "has_cva": int(has_cva),
        # If your model requires days_since_last_dressing_change
        # but you don't want to show it in the UI, set a default here:
        "days_since_last_dressing_change": 0,
        # Pass the *inverted* value so the model sees what it expects:
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

    df = pd.DataFrame([input_data])
    df_transformed = df.reindex(columns=model_feature_names, fill_value=0)

    prediction = model.predict(df_transformed)
    probability = model.predict_proba(df_transformed)[:, 1]

    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.header(f"Prediction: {risk_level}")
    st.subheader(f"Probability: {probability[0]:.2%}")
