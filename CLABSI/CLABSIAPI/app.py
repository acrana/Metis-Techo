import streamlit as st
import pandas as pd
import pickle
import json
import os


st.set_page_config(page_title="CLABSI Risk Prediction", layout="wide")

# Load model and features using relative paths
@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'final_xgb_model.pkl')
        features_path = os.path.join(current_dir, 'training_features.json')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(features_path, 'r') as f:
            features = json.load(f)
        return model, features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, TRAINING_FEATURES = load_model()

st.title('CLABSI Risk Prediction')

if model is None:
    st.error("Model failed to load. Please check file paths and permissions.")
else:
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
        
        df = pd.DataFrame([input_data])
        df_encoded = pd.get_dummies(df)
        df_transformed = df_encoded.reindex(columns=TRAINING_FEATURES, fill_value=0)
        
        prediction = model.predict(df_transformed)
        probability = model.predict_proba(df_transformed)[:, 1]
        
        risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
        st.header(f'Prediction: {risk_level}')
        st.subheader(f'Probability: {probability[0]:.2%}')
