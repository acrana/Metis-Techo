import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

if 'model1' not in st.session_state:
    model1 = joblib.load('clf1.pkl')
    model2 = joblib.load('clf2.pkl')
    model_clabsi = joblib.load('final_xgb_model.pkl')
    st.session_state.update({
        "model1": model1,
        "model2": model2,
        "model_clabsi": model_clabsi
    })
else:
    model1 = st.session_state["model1"]
    model2 = st.session_state["model2"]
    model_clabsi = st.session_state["model_clabsi"]

scaler1 = joblib.load("Scaler1.pkl")
scaler2 = joblib.load("Scaler2.pkl")
with open("training_features.json", "r") as f:
    TRAINING_FEATURES = json.load(f)

continuous_vars1 = ['LOS_before_using_IMV','LOS_before_using_CVC','APSIII','Temperature','LOS_before_using_IUC','MAP','PT']
continuous_vars2 = ['Age','Aniongap','APSIII','SAPII']

st.set_page_config(layout='wide')
st.write("<h1 style='text-align: center'>ICU Device Infection and Survival Prediction</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Device Infection & Survival", "CLABSI Prediction"])

with tab1:
    st.warning('The ML model assists ICU physicians in making informed decisions on optimal device insertion timing.')
    st.markdown('-----')
    dic2 = {'Yes': 1, 'No': 0}
    
    col1, col2 = st.columns(2)
    with col1:
        LOS_before_using_CVC = st.text_input("Length of stay before using CVC (hours)")
        LOS_before_using_IMV = st.text_input("Length of stay before using IMV (hours)")
        Tracheostomy = st.selectbox("Tracheostomy", ["Yes", "No"])
        APSIII = st.text_input("APSIII Score")
        MICU_or_SICU = st.selectbox("MICU or SICU", ["Yes", "No"])
        Temperature = st.text_input("Temperature (℃)")
        LOS_before_using_IUC = st.text_input("Length of stay before using IUC (hours)")
        MAP = st.text_input("Mean arterial pressure (mmHg)")
        RRT = st.selectbox("Renal replacement therapy", ["Yes", "No"])
        PT = st.text_input("PT (s)")

    with col2:
        Cancer = st.selectbox("Cancer", ["Yes", "No"])
        Age = st.text_input("Age (years)")    
        SAPII = st.text_input("SAPSII Score")    
        Cerebrovascular_disease = st.selectbox("Cerebrovascular disease", ["Yes", "No"])    
        Liver_disease = st.selectbox("Liver disease", ["Yes", "No"])    
        Aniongap = st.text_input("Anion gap (mmol/L)")     
        Myocardial_infarct = st.selectbox("Myocardial infarct", ["Yes", "No"])        
        Two_or_more_devices = st.selectbox("Using two or more devices", ["Yes", "No"])

    if st.button("Predict Device Infection & Survival"):
        with st.spinner("Forecasting..."):
            test_df = pd.DataFrame([float(LOS_before_using_CVC), float(LOS_before_using_IMV), dic2[Tracheostomy], 
                                  float(APSIII), dic2[MICU_or_SICU], float(Temperature), float(LOS_before_using_IUC), 
                                  float(MAP), dic2[RRT], float(PT)], 
                                 index=['LOS_before_using_CVC', 'LOS_before_using_IMV', 'Tracheostomy', 'APSIII', 
                                      'MICU_or_SICU', 'Temperature', 'LOS_before_using_IUC', 'MAP', 'RRT', 'PT']).T
            
            test_df[continuous_vars1] = scaler1.transform(test_df[continuous_vars1])
            infection_prob = model1.predict_proba(test_df)[:, 1][0]
            st.success(f"Probability of device-associated infection: {infection_prob * 100:.1f}%")
            
            st.header('30-day Survival Prediction')
            test_df2 = pd.DataFrame([dic2[MICU_or_SICU], dic2[Cancer], float(APSIII), float(Age), float(SAPII), 
                                   dic2[Cerebrovascular_disease], dic2[Liver_disease], float(Aniongap), 
                                   dic2[Myocardial_infarct], dic2[Two_or_more_devices]], 
                                  index=['MICU_or_SICU', 'Cancer', 'APSIII', 'Age', 'SAPII', 'Cerebrovascular_disease', 
                                       'Liver_disease', 'Aniongap', 'Myocardial_infarct', 'Two_or_more_devices']).T
            
            test_df2[continuous_vars2] = scaler2.transform(test_df2[continuous_vars2])
            surv_funcs = model2.predict_survival_function(test_df2)
            
            fig, ax = plt.subplots()
            for fn in surv_funcs:
                ax.step(fn.x, fn(fn.x), where="post", color="#8dd3c7", lw=2)
            plt.ylabel("Survival Probability (%)")
            plt.xlabel("Days since first invasive procedure")
            plt.grid()
            st.pyplot(fig)

with tab2:
    st.header("CLABSI Risk Prediction")
    st.warning("This model predicts CLABSI risk based on patient characteristics and averaged lab values over the line duration.")
    
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

    if st.button('Predict CLABSI Risk'):
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
        df_encoded = pd.get_dummies(df).reindex(columns=TRAINING_FEATURES, fill_value=0)
        prediction = model_clabsi.predict(df_encoded)
        probability = model_clabsi.predict_proba(df_encoded)[:, 1]
        
        risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
        st.header(f'Prediction: {risk_level}')
        st.subheader(f'CLABSI Probability: {probability[0]:.1%}')
        
        # Feature importance from model
        feature_importance = pd.DataFrame({
            'Feature': TRAINING_FEATURES,
            'Importance': model_clabsi.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])
        plt.title('Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)
