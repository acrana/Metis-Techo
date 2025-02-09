import streamlit as st
import joblib
import pandas as pd
import json
import os

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "final_xgb_model.pkl")
features_path = os.path.join(base_path, "training_features.json")

# Load model
model = joblib.load(model_path)

# If you have a training_features.json, you can still use it,
# but your model actually has "days_since_last_dressing_change"
# So let's just rely on the model's own stored feature names:
model_feature_names = model.get_booster().feature_names

# Or load your training_features if you prefer, but make sure
# it actually lists "days_since_last_dressing_change" too.
# with open(features_path, "r") as f:
#     TRAINING_FEATURES = json.load(f)

st.title("Line Risk Prediction (with hidden days_since_last_dressing_change)")

# UI fields - no mention of days_since_last_dressing_change
line_age = st.number_input("Line Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
has_diabetes = st.checkbox("Has Diabetes")
has_cancer = st.checkbox("Has Cancer")
has_liver = st.checkbox("Has Liver Disease")
has_chf = st.checkbox("Has CHF")
has_cva = st.checkbox("Has CVA")
chg_adherence_ratio = st.slider("CHG Adherence Ratio", 0.0, 1.0, 0.5)
wbc_mean = st.number_input("WBC Mean", min_value=0.0)
plt_mean = st.number_input("Platelet Mean", min_value=0.0)
creat_mean = st.number_input("Creatinine Mean", min_value=0.0)
inr_mean = st.number_input("INR Mean", min_value=0.0)
pt_mean = st.number_input("PT Mean", min_value=0.0)
sofa_score = st.number_input("SOFA Score", min_value=0)
apsiii = st.number_input("APSIII Score", min_value=0)
sapsii = st.number_input("SAPSII Score", min_value=0)

if st.button("Predict"):
    # Supply the model's required fields, including the missing one
    input_data = {
        "admission_age": line_age,
        "gender": gender,
        "has_diabetes": int(has_diabetes),
        "has_cancer": int(has_cancer),
        "has_liver": int(has_liver),
        "has_chf": int(has_chf),
        "has_cva": int(has_cva),
        "days_since_last_dressing_change": 0,  # Provide a default
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

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Force numeric
    df["gender"] = df["gender"].astype(int)

    # Reindex using the model's own feature names to ensure perfect match
    df_transformed = df.reindex(columns=model_feature_names, fill_value=0)

    # Debug
    st.write("Model expects columns:", model_feature_names)
    st.write("We're sending columns:", df_transformed.columns.tolist())

    # Predict
    prediction = model.predict(df_transformed)
    probability = model.predict_proba(df_transformed)[:, 1]

    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.header(f"Prediction: {risk_level}")
    st.subheader(f"Probability: {probability[0]:.2%}")
