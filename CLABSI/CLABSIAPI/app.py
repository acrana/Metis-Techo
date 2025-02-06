import streamlit as st
import pandas as pd
import pickle
import joblib
import json
import os
import matplotlib.pyplot as plt
import shap

# Set up page configuration
st.set_page_config(page_title="Combined CLABSI & Survival Prediction", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# LOAD MODELS & SCALERS
# ---------------------------
# For the infection risk model (Model 1)
MODEL1_PATH = os.path.join(BASE_DIR, 'clf1.pkl')
SCALER1_PATH = os.path.join(BASE_DIR, 'Scaler1.pkl')
CONTINUOUS_VARS1 = ['LOS_before_using_IMV','LOS_before_using_CVC','APSIII','Temperature','LOS_before_using_IUC','MAP','PT']

# For the survival model (Model 2)
MODEL2_PATH = os.path.join(BASE_DIR, 'clf2.pkl')
SCALER2_PATH = os.path.join(BASE_DIR, 'Scaler2.pkl')
CONTINUOUS_VARS2 = ['Age','Aniongap','APSIII','SAPII']

# Use st.session_state to cache loaded models
if 'model1' not in st.session_state:
    try:
        model1 = joblib.load(MODEL1_PATH)
        model2 = joblib.load(MODEL2_PATH)
        scaler1 = joblib.load(SCALER1_PATH)
        scaler2 = joblib.load(SCALER2_PATH)
        st.session_state['model1'] = model1
        st.session_state['model2'] = model2
        st.session_state['scaler1'] = scaler1
        st.session_state['scaler2'] = scaler2
    except Exception as e:
        st.error(f"Error loading models or scalers: {e}")
else:
    model1 = st.session_state['model1']
    model2 = st.session_state['model2']
    scaler1 = st.session_state['scaler1']
    scaler2 = st.session_state['scaler2']

# ---------------------------
# USER INTERFACE
# ---------------------------
st.title("Device-Associated Infection & 30-day Survival Prediction")

st.markdown("### Input Variables")
with st.sidebar:
    st.markdown("#### Infection Risk Inputs")
    LOS_before_using_CVC = st.text_input("ICU Length of Stay before using CVC (hours)", "0")
    LOS_before_using_IMV = st.text_input("ICU Length of Stay before using IMV (hours)", "0")
    Tracheostomy = st.selectbox("Tracheostomy", ["Yes", "No"])
    APSIII = st.text_input("APSIII (within 24 hrs of ICU admission)", "0")
    MICU_or_SICU = st.selectbox("Medical ICU or Surgical ICU", ["Yes", "No"])
    Temperature = st.text_input("Temperature (℃ within 24 hrs)", "37")
    LOS_before_using_IUC = st.text_input("ICU Length of Stay before using IUC (hours)", "0")
    MAP = st.text_input("Mean Arterial Pressure (mmHg within 24 hrs)", "70")
    RRT = st.selectbox("Renal Replacement Therapy", ["Yes", "No"])
    PT = st.text_input("Partial Thromboplastin Time (s within 24 hrs)", "30")
    
    st.markdown("#### Survival Prediction Inputs")
    Cancer = st.selectbox("Cancer", ["Yes", "No"])
    Age = st.text_input("Age (years)", "60")
    SAPII = st.text_input("SAPII (within 24 hrs of ICU admission)", "0")
    Cerebrovascular_disease = st.selectbox("Cerebrovascular Disease", ["Yes", "No"])
    Liver_disease = st.selectbox("Liver Disease", ["Yes", "No"])
    Aniongap = st.text_input("Anion Gap (mmol/L within 24 hrs)", "0")
    Myocardial_infarct = st.selectbox("Myocardial Infarct", ["Yes", "No"])
    Two_or_more_devices = st.selectbox("Using Two or More Devices", ["Yes", "No"])
    
# A simple dictionary to convert Yes/No inputs to binary values
dic = {"Yes": 1, "No": 0}

# ---------------------------
# PREDICTION BUTTON
# ---------------------------
if st.sidebar.button("Predict"):
    with st.spinner("Forecasting, please wait..."):
        # ---------------------------
        # Prepare infection risk input DataFrame for Model 1
        # ---------------------------
        try:
            data_infection = {
                'LOS_before_using_CVC': float(LOS_before_using_CVC),
                'LOS_before_using_IMV': float(LOS_before_using_IMV),
                'Tracheostomy': dic[Tracheostomy],
                'APSIII': float(APSIII),
                'MICU_or_SICU': dic[MICU_or_SICU],
                'Temperature': float(Temperature),
                'LOS_before_using_IUC': float(LOS_before_using_IUC),
                'MAP': float(MAP),
                'RRT': dic[RRT],
                'PT': float(PT)
            }
            df_infection = pd.DataFrame([data_infection])
        except Exception as e:
            st.error(f"Error processing infection risk inputs: {e}")
            st.stop()
            
        # Scale continuous variables for infection risk
        try:
            df_infection[CONTINUOUS_VARS1] = scaler1.transform(df_infection[CONTINUOUS_VARS1])
        except Exception as e:
            st.error(f"Error during scaling for infection risk model: {e}")
            st.stop()

        # ---------------------------
        # SHAP Explanation for Model 1
        # ---------------------------
        try:
            explainer = shap.Explainer(model1)
            shap_values = explainer(df_infection)
            # Plot a waterfall plot for the first instance
            fig1 = plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            shap_waterfall_path = os.path.join(BASE_DIR, 'shap_waterfall.png')
            plt.savefig(shap_waterfall_path, dpi=300)
            
            # Also create a force plot (if desired)
            fig2 = plt.figure()
            shap.force_plot(explainer.expected_value, shap_values.values[0, :], df_infection.iloc[0, :],
                            matplotlib=True, show=False, figsize=(20, 5))
            plt.tight_layout()
            shap_force_path = os.path.join(BASE_DIR, 'shap_force.png')
            plt.savefig(shap_force_path, dpi=300)
        except Exception as e:
            st.error(f"Error generating SHAP plots: {e}")
        
        # ---------------------------
        # Infection Risk Prediction
        # ---------------------------
        try:
            infection_prob = model1.predict_proba(df_infection)[:, 1][0]
            st.header("Prediction for Device-Associated Infection")
            st.success("Probability of infection: {:.3f}%".format(infection_prob * 100))
        except Exception as e:
            st.error(f"Error during infection risk prediction: {e}")
        
        # ---------------------------
        # Display SHAP plots for infection risk model
        # ---------------------------
        col1, col2 = st.columns(2)
        with col1:
            st.image(shap_waterfall_path, caption="SHAP Waterfall Plot")
        with col2:
            st.image(shap_force_path, caption="SHAP Force Plot")
        
        # ---------------------------
        # Prepare survival prediction DataFrame for Model 2
        # ---------------------------
        try:
            data_survival = {
                'MICU_or_SICU': dic[MICU_or_SICU],
                'Cancer': dic[Cancer],
                'APSIII': float(APSIII),
                'Age': float(Age),
                'SAPII': float(SAPII),
                'Cerebrovascular_disease': dic[Cerebrovascular_disease],
                'Liver_disease': dic[Liver_disease],
                'Aniongap': float(Aniongap),
                'Myocardial_infarct': dic[Myocardial_infarct],
                'Two_or_more_devices': dic[Two_or_more_devices]
            }
            df_survival = pd.DataFrame([data_survival])
        except Exception as e:
            st.error(f"Error processing survival inputs: {e}")
            st.stop()
            
        # Scale continuous variables for survival model
        try:
            df_survival[CONTINUOUS_VARS2] = scaler2.transform(df_survival[CONTINUOUS_VARS2])
        except Exception as e:
            st.error(f"Error during scaling for survival model: {e}")
            st.stop()
        
        # ---------------------------
        # Survival Prediction (Kaplan-Meier)
        # ---------------------------
        try:
            surv_funcs = model2.predict_survival_function(df_survival)
            st.header("30-day Kaplan-Meier Survival Curve")
            fig_surv, ax_surv = plt.subplots()
            for fn in surv_funcs:
                # Plot survival probability over time
                ax_surv.step(fn.x, fn(fn.x), where="post", color="#8dd3c7", lw=2)
            ax_surv.set_ylabel("Probability of survival (%)")
            ax_surv.set_xlabel("Time since invasive procedure (days)")
            ax_surv.grid(True)
            st.pyplot(fig_surv)
        except Exception as e:
            st.error(f"Error during survival prediction: {e}")

