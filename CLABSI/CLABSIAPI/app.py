import streamlit as st
import joblib
import pandas as pd
import json
import os

# --------------------------------------------------------------------
# 1. Locate directory of this file and build absolute paths
# --------------------------------------------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, "final_xgb_model.pkl")
features_path = os.path.join(base_path, "training_features.json")

# --------------------------------------------------------------------
# 2. Load the model and features
# --------------------------------------------------------------------
model = joblib.load(model_path)
with open(features_path, "r") as f:
    TRAINING_FEATURES = json.load(f)

st.title("Line Risk Prediction")

# Two-column layout in Streamlit
col1, col2 = st.columns(2)

with col1:
    # Show "Line Age" in the UI, store in variable line_age
    line_age = st.number_input("Line Age", min_value=0, max_value=120)
    # gender is numeric: 0 for female, 1 for male
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

# --------------------------------------------------------------------
# 3. When user clicks "Predict"
# --------------------------------------------------------------------
if st.button("Predict"):
    # Create input data using the keys your model expects
    input_data = {
        "admission_age": line_age,  # Keep the internal key as "admission_age"
        "gender": gender,
        "has_diabetes": int(has_diabetes),
        "has_cancer": int(has_cancer),
        "has_liver": int(has_liver),
        "has_chf": int(has_chf),
        "has_cva": int(has_cva),
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

    # ---------------------------------------------------------------
    # If your training used strictly numeric columns (0 or 1 for gender),
    # you don't need get_dummies. We'll skip it.
    # ---------------------------------------------------------------
    # Just ensure "gender" is int (0,1).
    df["gender"] = df["gender"].astype(int)

    # ---------------------------------------------------------------
    # Reindex to match your training feature list exactly
    # ---------------------------------------------------------------
    df_transformed = df.reindex(columns=TRAINING_FEATURES, fill_value=0)

    # DEBUG: Print out the columns to see what's going to the model
    st.write("Columns in final df_transformed:", df_transformed.columns.tolist())

    # ---------------------------------------------------------------
    # 4. Predict with XGBoost
    # ---------------------------------------------------------------
    prediction = model.predict(df_transformed)
    probability = model.predict_proba(df_transformed)[:, 1]

    # Map the prediction
    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"

    st.header(f"Prediction: {risk_level}")
    st.subheader(f"Probability: {probability[0]:.2%}")
