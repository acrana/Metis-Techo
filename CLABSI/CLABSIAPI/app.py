import streamlit as st
import joblib
import pandas as pd
import json

# Load model and features
model = joblib.load('final_xgb_model.pkl')
with open("training_features.json", "r") as f:
    TRAINING_FEATURES = json.load(f)

st.title('Line Risk Prediction')

# Input form
col1, col2 = st.columns(2)

with col1:
    admission_age = st.number_input('Admission Age', min_value=0, max_value=120)
    gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x==0 else 'Male')
    has_diabetes = st.checkbox('Has Diabetes')
    has_cancer = st.checkbox('Has Cancer')
    has_liver = st.checkbox('Has Liver Disease')
    has_chf = st.checkbox('Has CHF')
    has_cva = st.checkbox('Has CVA')
    days_since_last_dressing_change = st.number_input('Days Since Last Dressing Change', min_value=0)
    chg_adherence_ratio = st.slider('CHG Adherence Ratio', 0.0, 1.0, 0.5)

with col2:
    wbc_mean = st.number_input('WBC Mean', min_value=0.0)
    plt_mean = st.number_input('Platelet Mean', min_value=0.0)
    creat_mean = st.number_input('Creatinine Mean', min_value=0.0)
    inr_mean = st.number_input('INR Mean', min_value=0.0)
    pt_mean = st.number_input('PT Mean', min_value=0.0)
    sofa_score = st.number_input('SOFA Score', min_value=0)
    apsiii = st.number_input('APSIII Score', min_value=0)
    sapsii = st.number_input('SAPSII Score', min_value=0)

if st.button('Predict'):
    # Create input data dictionary
    input_data = {
        'admission_age': admission_age,
        'gender': gender,
        'has_diabetes': int(has_diabetes),
        'has_cancer': int(has_cancer),
        'has_liver': int(has_liver),
        'has_chf': int(has_chf),
        'has_cva': int(has_cva),
        'days_since_last_dressing_change': days_since_last_dressing_change,
        'chg_adherence_ratio': chg_adherence_ratio,
        'wbc_mean': wbc_mean,
        'plt_mean': plt_mean,
        'creat_mean': creat_mean,
        'inr_mean': inr_mean,
        'pt_mean': pt_mean,
        'sofa_score': sofa_score,
        'apsiii': apsiii,
        'sapsii': sapsii
    }

    # Transform data
    df = pd.DataFrame([input_data])
    df_encoded = pd.get_dummies(df)
    df_transformed = df_encoded.reindex(columns=TRAINING_FEATURES, fill_value=0)

    # Get prediction
    prediction = model.predict(df_transformed)
    probability = model.predict_proba(df_transformed)[:, 1]

    # Display results
    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.header(f'Prediction: {risk_level}')
    st.subheader(f'Probability: {probability[0]:.2%}')
