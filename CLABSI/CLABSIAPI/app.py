import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import json
import os

# Load all models
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

# Load scalers and features
scaler1 = joblib.load("Scaler1.pkl")
scaler2 = joblib.load("Scaler2.pkl")
with open("training_features.json", "r") as f:
    TRAINING_FEATURES = json.load(f)

continuous_vars1 = ['LOS_before_using_IMV','LOS_before_using_CVC','APSIII','Temperature','LOS_before_using_IUC','MAP','PT']
continuous_vars2 = ['Age','Aniongap','APSIII','SAPII']

st.set_page_config(layout='wide')

st.write("<h1 style='text-align: center'>ICU Device Infection and Survival Prediction</h1>", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["Device Infection & Survival", "CLABSI Prediction"])

with tab1:
    st.warning('The ML model assists ICU physicians in making informed decisions on optimal device insertion timing to reduce the risk of device-associated infections.')

    st.markdown('-----')
    dic2 = {
        'Yes': 1,
        'No': 0
    }
    
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
            st.header('Device-associated Infection Prediction')
            test_df = pd.DataFrame([float(LOS_before_using_CVC), float(LOS_before_using_IMV), dic2[Tracheostomy], 
                                  float(APSIII), dic2[MICU_or_SICU], float(Temperature), float(LOS_before_using_IUC), 
                                  float(MAP), dic2[RRT], float(PT)], 
                                 index=['LOS_before_using_CVC', 'LOS_before_using_IMV', 'Tracheostomy', 'APSIII', 
                                      'MICU_or_SICU', 'Temperature', 'LOS_before_using_IUC', 'MAP', 'RRT', 'PT']).T
            
            test_df[continuous_vars1] = scaler1.transform(test_df[continuous_vars1])
            explainer = shap.Explainer(model1)
            shap_ = explainer.shap_values(test_df)
            
            shap.waterfall_plot(
                shap.Explanation(values=shap_[0, :], base_values=explainer.expected_value, data=test_df.iloc[0, :]),
                show=False)
            plt.tight_layout()
            plt.savefig('shap1.png', dpi=300)
            
            shap.initjs()
            shap.force_plot(explainer.expected_value, shap_[0, :], test_df.iloc[0, :], show=False, matplotlib=True,
                            figsize=(20, 5))
            plt.tight_layout()
            plt.savefig('shap2.png', dpi=300)
            
            col1, col2, col3 = st.columns([2, 5, 3])
            with col2:
                st.image('shap1.png')
                st.image('shap2.png')
            
            st.success(f"Probability of device-associated infection: {model1.predict_proba(test_df)[:, 1][0] * 100:.1f}%")
            
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
            
            test_prob = model_clabsi.predict_proba(test_encoded)[:, 1][0]
            baseline_prob = model_clabsi.predict_proba(baseline_encoded)[:, 1][0]
            
            impact_scores[feature] = test_prob - baseline_prob

        df_encoded = pd.get_dummies(df).reindex(columns=TRAINING_FEATURES, fill_value=0)
        prediction = model_clabsi.predict(df_encoded)
        probability = model_clabsi.predict_proba(df_encoded)[:, 1]
        
        risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
        st.header(f'Prediction: {risk_level}')
        st.subheader(f'CLABSI Probability: {probability[0]:.1%}')
        
        impacts = pd.DataFrame({
            'Feature': list(impact_scores.keys()),
            'Impact': list(impact_scores.values())
        }).sort_values('Impact', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if x > 0 else 'green' for x in impacts['Impact']]
        plt.barh(impacts['Feature'], impacts['Impact'], color=colors)
        plt.title('Feature Impact on CLABSI Risk')
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
        
        if not top_decreasing.empty:
            st.write("Factors decreasing risk:")
            for _, row in top_decreasing.iterrows():
                st.write(f"• {row['Feature']}: {row['Impact']*100:.1f}% risk")
