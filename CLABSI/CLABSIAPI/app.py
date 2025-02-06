import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# Get current directory and build paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'final_xgb_model.pkl')
features_path = os.path.join(current_dir, 'training_features.json')

# Debug file paths
st.write("Current directory:", current_dir)
st.write("Model exists:", os.path.exists(model_path))
st.write("Features exists:", os.path.exists(features_path))

# Load CLABSI model
if 'model_clabsi' not in st.session_state:
    model_clabsi = joblib.load('final_xgb_model.pkl')
    st.session_state["model_clabsi"] = model_clabsi
else:
    model_clabsi = st.session_state["model_clabsi"]

with open("training_features.json", "r") as f:
    TRAINING_FEATURES = json.load(f)

st.set_page_config(layout='wide')
st.title('CLABSI Risk Prediction')

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
    
    # Calculate impacts
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
        
        test_prob = model_clabsi.predict_proba(test_encoded)[:, 1][0]
        baseline_prob = model_clabsi.predict_proba(baseline_encoded)[:, 1][0]
        
        impact_scores[feature] = test_prob - baseline_prob

    df_encoded = pd.get_dummies(df).reindex(columns=TRAINING_FEATURES, fill_value=0)
    prediction = model_clabsi.predict(df_encoded)
    probability = model_clabsi.predict_proba(df_encoded)[:, 1]
    
    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.header(f'Prediction: {risk_level}')
    st.subheader(f'Probability: {probability[0]:.1%}')
    
    impacts = pd.DataFrame({
        'Feature': list(impact_scores.keys()),
        'Impact': list(impact_scores.values())
    }).sort_values('Impact', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x > 0 else 'green' for x in impacts['Impact']]
    plt.barh(impacts['Feature'], impacts['Impact'], color=colors)
    plt.title('Feature Impact on Risk (Red = Increases Risk, Green = Decreases Risk)')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)

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
