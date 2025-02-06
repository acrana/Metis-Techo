import streamlit as st
import pandas as pd
import pickle
import json
import os
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="CLABSI Risk Prediction", layout="wide")

# Get absolute path to the directory containing your files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'final_xgb_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'training_features.json')

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(FEATURES_PATH, 'r') as f:
            features = json.load(f)
        return model, features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}\nTried path: {MODEL_PATH}")
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

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_transformed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        # Create feature importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance = pd.DataFrame({
            'Feature': df_transformed.columns,
            'Importance': abs(shap_values[0])
        }).sort_values('Importance', ascending=True)
        
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])
        plt.title('Feature Importance (SHAP values)')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show top contributing factors
        st.subheader("Top Contributing Factors:")
        top_features = feature_importance.tail(5)
        for idx, row in top_features.iterrows():
            value = df_transformed.iloc[0][row['Feature']]
            impact = "increasing" if shap_values[0][idx] > 0 else "decreasing"
            st.write(f"{row['Feature']} (value: {value:.2f}): {impact} risk by {abs(shap_values[0][idx]):.3f}")
