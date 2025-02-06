import streamlit as st
import pandas as pd
import pickle
import json
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="CLABSI Risk Prediction", layout="wide")

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
        admission_age = st.number_input('Line Age (Days)', min_value=0, max_value=120)
        gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x==0 else 'Male')
        has_diabetes = st.checkbox('Has Diabetes')
        has_cancer = st.checkbox('Has Cancer')
        has_liver = st.checkbox('Has Liver Disease')
        has_chf = st.checkbox('Has CHF')
        has_cva = st.checkbox('Has CVA')
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
        
        # Calculate impact scores
        impact_scores = {}
        baseline_values = {
            'admission_age': 10,
            'gender': 0,
            'has_diabetes': 0,
            'has_cancer': 0,
            'has_liver': 0,
            'has_chf': 0,
            'has_cva': 0,
            'chg_adherence_ratio': 0.9,
            'wbc_mean': 7.5,
            'plt_mean': 150,
            'creat_mean': 1.0,
            'inr_mean': 1.0,
            'pt_mean': 11,
            'sofa_score': 0,
            'apsiii': 40,
            'sapsii': 30
        }
        
        for feature, value in input_data.items():
            baseline = baseline_values[feature]
            
            test_df = df.copy()
            baseline_df = df.copy()
            baseline_df[feature] = baseline
            
            test_encoded = pd.get_dummies(test_df).reindex(columns=TRAINING_FEATURES, fill_value=0)
            baseline_encoded = pd.get_dummies(baseline_df).reindex(columns=TRAINING_FEATURES, fill_value=0)
            
            test_prob = model.predict_proba(test_encoded)[:, 1][0]
            baseline_prob = model.predict_proba(baseline_encoded)[:, 1][0]
            
            impact_scores[feature] = test_prob - baseline_prob

        # Original prediction
        df_encoded = pd.get_dummies(df).reindex(columns=TRAINING_FEATURES, fill_value=0)
        prediction = model.predict(df_encoded)
        probability = model.predict_proba(df_encoded)[:, 1]
        
        # Display results
        risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
        st.header(f'Prediction: {risk_level}')
        st.subheader(f'Probability: {probability[0]:.2%}')
        
        # Sort and display impacts
        impacts = pd.DataFrame({
            'Feature': list(impact_scores.keys()),
            'Impact': list(impact_scores.values())
        }).sort_values('Impact', ascending=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if x > 0 else 'green' for x in impacts['Impact']]
        plt.barh(impacts['Feature'], impacts['Impact'], color=colors)
        plt.title('Feature Impact on Risk (Red = Increases Risk, Green = Decreases Risk)')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        # Show top factors
        st.subheader("Key Risk Factors:")
        top_increasing = impacts[impacts['Impact'] > 0].tail(3)
        top_decreasing = impacts[impacts['Impact'] < 0].head(3)
        
        if not top_increasing.empty:
            st.write("Factors increasing risk:")
            for _, row in top_increasing.iterrows():
                st.write(f"• {row['Feature']}: +{row['Impact']*100:.1f}% risk")
        
        if not top_decreasing.empty:
            st.write("Factors decreasing risk:")
            for _, row in top_decreasing.iterrows():
                st.write(f"• {row['Feature']}: {row['Impact']*100:.1f}% risk")
