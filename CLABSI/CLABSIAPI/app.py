import streamlit as st
import pandas as pd
import pickle
import json
import os
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn import metrics

# -------------------------------------------------------------
# 1. SET STREAMLIT PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="Combined CLABSI Prediction App", layout="wide")

# -------------------------------------------------------------
# 2. LOAD ALL MODELS & FEATURES
# -------------------------------------------------------------
@st.cache_resource
def load_models_and_features():
    """
    Loads and returns all models & supporting objects.
    We do it in one place to keep the code cleaner.
    """
    # 2.1. XGB model + features
    xgb_model_path = "final_xgb_model.pkl"
    xgb_features_path = "training_features.json"
    xgb_model = None
    xgb_features = None

    # Attempt to load the XGB model
    try:
        with open(xgb_model_path, 'rb') as f:
            xgb_model = pickle.load(f)
        with open(xgb_features_path, 'r') as f:
            xgb_features = json.load(f)
    except Exception as e:
        st.error(f"Could not load XGB model or features: {e}")

    # 2.2. joblib-based models (model1, model2) + scalers
    # If you have them in session_state already, fine. Otherwise, load them from file:
    try:
        model1 = joblib.load("clf1.pkl")
        model2 = joblib.load("clf2.pkl")
        scaler1 = joblib.load("Scaler1.pkl")
        scaler2 = joblib.load("Scaler2.pkl")
    except Exception as e:
        st.error(f"Could not load joblib-based models or scalers: {e}")
        model1, model2, scaler1, scaler2 = None, None, None, None

    # Return everything
    return xgb_model, xgb_features, model1, model2, scaler1, scaler2


# Actually load everything
xgb_model, xgb_features, model1, model2, scaler1, scaler2 = load_models_and_features()

# For reference, your continuous variable lists:
continuous_vars1 = [
    'LOS_before_using_IMV','LOS_before_using_CVC','APSIII','Temperature',
    'LOS_before_using_IUC','MAP','PT'
]
continuous_vars2 = [
    'Age','Aniongap','APSIII','SAPII'
]

# -------------------------------------------------------------
# 3. SIDEBAR TO CHOOSE WHICH MODEL OR PAGE
# -------------------------------------------------------------
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Choose a model or page:",
    ["XGB CLABSI Risk", "Joblib Infection + Survival"]
)

# -------------------------------------------------------------
# 4. PAGE 1: XGB CLABSI RISK
# -------------------------------------------------------------
if page_selection == "XGB CLABSI Risk":
    st.title("CLABSI Risk Prediction (XGB)")

    if xgb_model is None or xgb_features is None:
        st.error("XGB model or feature list not loaded correctly. Check file paths.")
    else:
        # --- UI for XGB inputs ---
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

        if st.button('Predict with XGB'):
            # Prepare the data
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

            # For "feature impact" logic, define your baseline values:
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

            # Compute impacts
            impact_scores = {}
            for feature, value in input_data.items():
                baseline = baseline_values[feature]

                # make copies
                test_df = df.copy()
                baseline_df = df.copy()
                baseline_df[feature] = baseline

                # convert to dummies with training features
                test_encoded = pd.get_dummies(test_df).reindex(columns=xgb_features, fill_value=0)
                baseline_encoded = pd.get_dummies(baseline_df).reindex(columns=xgb_features, fill_value=0)

                test_prob = xgb_model.predict_proba(test_encoded)[:, 1][0]
                baseline_prob = xgb_model.predict_proba(baseline_encoded)[:, 1][0]
                impact_scores[feature] = test_prob - baseline_prob

            # Original prediction
            df_encoded = pd.get_dummies(df).reindex(columns=xgb_features, fill_value=0)
            prediction = xgb_model.predict(df_encoded)
            probability = xgb_model.predict_proba(df_encoded)[:, 1]

            # Display results
            risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
            st.header(f'Prediction: {risk_level}')
            st.subheader(f'Probability: {probability[0]:.2%}')

            # Sort and display impacts
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


# -------------------------------------------------------------
# 5. PAGE 2: JOBLIB INFECTION + SURVIVAL
# -------------------------------------------------------------
elif page_selection == "Joblib Infection + Survival":
    st.title("Device-Associated Infection Prediction + 30-day Survival")
    st.warning(
        "The ML model assists ICU physicians in making informed decisions "
        "on optimal device insertion timing to reduce the risk of device-associated infections."
    )
    st.markdown('---')

    if model1 is None or model2 is None or scaler1 is None or scaler2 is None:
        st.error("Joblib-based models or scalers not loaded correctly.")
    else:
        # -- Sidebar or main bar for inputs --
        st.markdown("#### Input Variables")
        LOS_before_using_CVC = st.text_input(
            "Length of stay in the ICU (hours) before using the **CVC**",
            value="0"
        )
        LOS_before_using_IMV = st.text_input(
            "Length of stay in the ICU (hours) before using the **IMV**",
            value="0"
        )
        Tracheostomy = st.selectbox("Tracheostomy", ["Yes", "No"])
        APSIII_1 = st.text_input("APSIII within 24 hours of ICU admission", value="0")
        MICU_or_SICU = st.selectbox("Medical or Surgical ICU", ["Yes", "No"])
        Temperature = st.text_input("Temperature (℃) within 24 hours of ICU admission", value="36.5")
        LOS_before_using_IUC = st.text_input(
            "Length of stay in the ICU (hours) before using the **IUC**",
            value="0"
        )
        MAP = st.text_input("Mean arterial pressure (mmHg) within 24 hours of ICU admission", value="70")
        RRT = st.selectbox("Renal replacement therapy", ["Yes", "No"])
        PT = st.text_input("Partial thromboplastin time (s) within 24 hours of ICU admission", value="11")
        Cancer = st.selectbox("Cancer", ["Yes", "No"])
        Age = st.text_input("Age (years)", value="60")
        SAPII = st.text_input("SAPII within 24 hours of ICU admission", value="40")
        Cerebrovascular_disease = st.selectbox("Cerebrovascular disease", ["Yes", "No"])
        Liver_disease = st.selectbox("Liver disease", ["Yes", "No"])
        Aniongap = st.text_input("Anion gap (mmol/L) within 24 hours of ICU admission", value="12")
        Myocardial_infarct = st.selectbox("Myocardial infarct", ["Yes", "No"])
        Two_or_more_devices = st.selectbox("Using two or more devices", ["Yes", "No"])

        # map for yes/no
        dic2 = {'Yes': 1, 'No': 0}

        if st.button("Predict with Joblib Models"):
            with st.spinner("Working..."):
                # ----------------------------------------------------
                # 5.1 Infection Risk (model1) with shap
                # ----------------------------------------------------
                st.header("Prediction for Device-Associated Infections")

                # Build a one-row DataFrame for model1
                # These features match what model1 expects:
                df1 = pd.DataFrame([[
                    float(LOS_before_using_CVC),
                    float(LOS_before_using_IMV),
                    dic2[Tracheostomy],
                    float(APSIII_1),
                    dic2[MICU_or_SICU],
                    float(Temperature),
                    float(LOS_before_using_IUC),
                    float(MAP),
                    dic2[RRT],
                    float(PT)
                ]], columns=[
                    'LOS_before_using_CVC',
                    'LOS_before_using_IMV',
                    'Tracheostomy',
                    'APSIII',
                    'MICU_or_SICU',
                    'Temperature',
                    'LOS_before_using_IUC',
                    'MAP',
                    'RRT',
                    'PT'
                ])

                # Scale continuous features
                df1[continuous_vars1] = scaler1.transform(df1[continuous_vars1])

                # SHAP explanations
                explainer = shap.Explainer(model1)
                shap_values = explainer.shap_values(df1)

                # Waterfall Plot
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0, :],
                        base_values=explainer.expected_value,
                        data=df1.iloc[0, :]
                    ),
                    show=False
                )
                plt.tight_layout()
                plt.savefig('shap_infection_waterfall.png', dpi=300)

                # Force Plot
                shap.initjs()
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0, :],
                    df1.iloc[0, :],
                    show=False,
                    matplotlib=True,
                    figsize=(20,5)
                )
                plt.xticks(fontproperties='Times New Roman', size=16)
                plt.yticks(fontproperties='Times New Roman', size=16)
                plt.tight_layout()
                plt.savefig('shap_infection_force.png', dpi=300)

                colA, colB, colC = st.columns([2, 5, 3])
                with colB:
                    st.image('shap_infection_waterfall.png')
                    st.image('shap_infection_force.png')

                infection_prob = model1.predict_proba(df1)[:, 1][0] * 100
                st.success(f"Probability of device-associated infection: {infection_prob:.3f}%")

                # ----------------------------------------------------
                # 5.2 30-day Survival (model2)
                # ----------------------------------------------------
                st.header("30-day Kaplan-Meier Survival Curve")

                # Build DataFrame for model2
                df2 = pd.DataFrame([[
                    dic2[MICU_or_SICU],
                    dic2[Cancer],
                    float(APSIII_1),
                    float(Age),
                    float(SAPII),
                    dic2[Cerebrovascular_disease],
                    dic2[Liver_disease],
                    float(Aniongap),
                    dic2[Myocardial_infarct],
                    dic2[Two_or_more_devices]
                ]], columns=[
                    'MICU_or_SICU',
                    'Cancer',
                    'APSIII',
                    'Age',
                    'SAPII',
                    'Cerebrovascular_disease',
                    'Liver_disease',
                    'Aniongap',
                    'Myocardial_infarct',
                    'Two_or_more_devices'
                ])

                df2[continuous_vars2] = scaler2.transform(df2[continuous_vars2])

                # model2 should be a survival model with .predict_survival_function
                surv_funcs = model2.predict_survival_function(df2)

                fig, ax = plt.subplots()
                for fn in surv_funcs:
                    ax.step(fn.x, fn(fn.x), where="post", color="#8dd3c7", lw=2)
                plt.ylabel("Probability of survival (%)")
                plt.xlabel("Time since first invasive procedure (days)")
                plt.grid()

                st.pyplot(fig)

                st.success("Survival function plotted successfully!")


# End of app

