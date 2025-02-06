import streamlit as st
import pandas as pd
import pickle
import json
import os
import matplotlib.pyplot as plt
import joblib  # to load clf1.pkl

st.set_page_config(page_title="CLABSI Risk Prediction", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Original XGB model paths:
MODEL_PATH = os.path.join(BASE_DIR, 'final_xgb_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'training_features.json')

# OPTIONAL: second model path
CLF1_PATH = os.path.join(BASE_DIR, 'clf1.pkl')


@st.cache_resource
def load_models():
    """Load both the XGB model/features AND a second model (if you want)."""
    xgb_model = None
    xgb_features = None
    clf1_model = None

    # Load XGB model
    try:
        with open(MODEL_PATH, 'rb') as f:
            xgb_model = pickle.load(f)
        with open(FEATURES_PATH, 'r') as f:
            xgb_features = json.load(f)
    except Exception as e:
        st.error(f"Error loading XGB model: {str(e)}")

    # Load second model
    try:
        clf1_model = joblib.load(CLF1_PATH)
    except Exception as e:
        st.warning(f"Could not load clf1.pkl: {str(e)}")

    return xgb_model, xgb_features, clf1_model


xgb_model, TRAINING_FEATURES, clf1 = load_models()

st.title('CLABSI Risk Prediction')

# --- If XGB model is missing, show error and exit
if xgb_model is None:
    st.error("XGB model failed to load. Please check file paths and permissions.")
    st.stop()

# --- UI for user inputs (SAME as your original)
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

    # ----------------------------------------------------------------
    # 1) XGB Prediction (same as your original code)
    # ----------------------------------------------------------------

    # Calculate impact scores
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

    impact_scores = {}
    for feature, value in input_data.items():
        baseline = baseline_values[feature]
        test_df = df.copy()
        baseline_df = df.copy()
        baseline_df[feature] = baseline

        test_encoded = pd.get_dummies(test_df).reindex(columns=TRAINING_FEATURES, fill_value=0)
        baseline_encoded = pd.get_dummies(baseline_df).reindex(columns=TRAINING_FEATURES, fill_value=0)

        test_prob = xgb_model.predict_proba(test_encoded)[:, 1][0]
        baseline_prob = xgb_model.predict_proba(baseline_encoded)[:, 1][0]
        impact_scores[feature] = test_prob - baseline_prob

    df_encoded = pd.get_dummies(df).reindex(columns=TRAINING_FEATURES, fill_value=0)
    prediction = xgb_model.predict(df_encoded)
    probability = xgb_model.predict_proba(df_encoded)[:, 1]

    # Display XGB results
    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.header(f'XGB Prediction: {risk_level}')
    st.subheader(f'XGB Probability: {probability[0]:.2%}')

    # Sort and plot impacts
    impacts_df = pd.DataFrame({
        'Feature': list(impact_scores.keys()),
        'Impact': list(impact_scores.values())
    }).sort_values('Impact', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x > 0 else 'green' for x in impacts_df['Impact']]
    plt.barh(impacts_df['Feature'], impacts_df['Impact'], color=colors)
    plt.title('Feature Impact on XGB Risk')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    # ----------------------------------------------------------------
    # 2) OPTIONAL: Make a small prediction using clf1 (if loaded)
    # ----------------------------------------------------------------
    if clf1 is not None:
        st.markdown("---")
        st.subheader("Secondary Model Prediction (clf1)")
        # For example, maybe we just do a .predict() on the same features:
        # NOTE: This depends on what clf1 actually expects as input.
        # If clf1 has different columns or a different pipeline, adapt accordingly.

        # We'll assume clf1 expects the same columns for demonstration:
        try:
            secondary_pred = clf1.predict(df_encoded)
            secondary_prob = clf1.predict_proba(df_encoded)[:, 1]

            st.write(f"clf1 Predicted Label: {secondary_pred[0]}")
            st.write(f"clf1 Predicted Probability: {secondary_prob[0]:.2%}")
        except Exception as e:
            st.error(f"Error predicting with clf1: {e}")

    else:
        st.warning("clf1 model not loaded or not found. (Check your .pkl path?)")

