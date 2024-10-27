# src/preprocessing.py

import sqlite3
import pandas as pd
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler

def load_data():
    # Database path
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)

    # Read tables into DataFrames
    demographics_df = pd.read_sql_query("SELECT * FROM TBL_Demographics", conn)
    survey_df = pd.read_sql_query("SELECT * FROM TBL_Survey", conn)
    ade_records_df = pd.read_sql_query("SELECT * FROM TBL_ADERecords", conn)
    prescriptions_df = pd.read_sql_query("SELECT * FROM TBL_Prescriptions", conn)
    medications_df = pd.read_sql_query("SELECT * FROM TBL_Medications", conn)

    conn.close()
    return demographics_df, survey_df, ade_records_df, prescriptions_df, medications_df

def merge_data(demographics_df, survey_df, ade_records_df, prescriptions_df, medications_df):
    # Merge demographics and survey data
    data = pd.merge(demographics_df, survey_df, on='PatientID', how='left')

    # Merge prescriptions with medications
    prescriptions_df = pd.merge(prescriptions_df, medications_df, on='MedicationID', how='left')

    # Merge with main data
    data = pd.merge(data, prescriptions_df[['PatientID', 'MedicationName', 'Date']], on='PatientID', how='left')

    # Create target variable
    patients_with_ade = ade_records_df['PatientID'].unique()
    data['Had_ADE'] = data['PatientID'].apply(lambda x: 1 if x in patients_with_ade else 0)

    # Handle missing medications (patients without prescriptions)
    data['MedicationName'] = data['MedicationName'].fillna('None')

    return data

def preprocess_data():
    demographics_df, survey_df, ade_records_df, prescriptions_df, medications_df = load_data()
    data = merge_data(demographics_df, survey_df, ade_records_df, prescriptions_df, medications_df)

    # Handle missing values
    data = data.dropna()

    # Encode categorical variables
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    # One-hot encode 'MedicationName'
    data = pd.get_dummies(data, columns=['MedicationName'], prefix='Med', drop_first=True)

    # Convert dates and calculate LengthOfStay
    data['AdmissionDate'] = pd.to_datetime(data['AdmissionDate'])
    data['DischargeDate'] = pd.to_datetime(data['DischargeDate'])
    data['LengthOfStay'] = (data['DischargeDate'] - data['AdmissionDate']).dt.days

    # Drop irrelevant columns
    data = data.drop(['PatientLast', 'PatientFirst', 'DOB', 'AdmissionDate', 'DischargeDate', 'AttributeName', 'SurveyDate', 'Date'], axis=1)

    # Feature columns
    feature_columns = [col for col in data.columns if col != 'Had_ADE']

    # Ensure consistent feature ordering
    data = data[feature_columns + ['Had_ADE']]

    # Save feature columns for future use
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    feature_columns_path = os.path.join(models_dir, 'features.json')
    with open(feature_columns_path, 'w') as f:
        json.dump(feature_columns, f)

    return data

def preprocess_individual_patient(patient_id, medication_name):
    # Load the patient's data
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)

    # Get patient demographics
    demographics_query = "SELECT * FROM TBL_Demographics WHERE PatientID = ?"
    demographics_df = pd.read_sql_query(demographics_query, conn, params=(patient_id,))

    # Get patient survey data
    survey_query = "SELECT * FROM TBL_Survey WHERE PatientID = ?"
    survey_df = pd.read_sql_query(survey_query, conn, params=(patient_id,))

    conn.close()

    if demographics_df.empty or survey_df.empty:
        print(f"No data found for PatientID {patient_id}.")
        return None

    # Merge demographics and survey data
    data = pd.merge(demographics_df, survey_df, on='PatientID', how='left')

    # Add medication data
    data['MedicationName'] = medication_name

    # Handle missing surveys or medications
    data = data.dropna()

    # Encode categorical variables
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    # One-hot encode 'MedicationName'
    data = pd.get_dummies(data, columns=['MedicationName'], prefix='Med', drop_first=True)

    # Convert dates and calculate LengthOfStay
    data['AdmissionDate'] = pd.to_datetime(data['AdmissionDate'])
    data['DischargeDate'] = pd.to_datetime(data['DischargeDate'])
    data['LengthOfStay'] = (data['DischargeDate'] - data['AdmissionDate']).dt.days

    # Drop irrelevant columns
    data = data.drop(['PatientLast', 'PatientFirst', 'DOB', 'AdmissionDate', 'DischargeDate', 'AttributeName', 'SurveyDate'], axis=1)

    # Load feature columns
    feature_columns_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'features.json')
    if not os.path.exists(feature_columns_path):
        print("Feature columns not found. Please ensure you've saved the feature columns during model training.")
        return None
    with open(feature_columns_path, 'r') as f:
        feature_columns = json.load(f)

    # Reindex to ensure all feature columns are present
    data = data.reindex(columns=feature_columns, fill_value=0)

    # Load the scaler
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.joblib')
    if not os.path.exists(scaler_path):
        print("Scaler not found. Please ensure you've saved the scaler during model training.")
        return None
    scaler = joblib.load(scaler_path)

    # Scale the features
    features_scaled = scaler.transform(data)

    return features_scaled

