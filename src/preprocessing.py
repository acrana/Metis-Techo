import sqlite3
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

def load_data():
    # Database path
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)

    # Read tables into DataFrames
    demographics_df = pd.read_sql_query("SELECT * FROM TBL_Demographics", conn)
    survey_df = pd.read_sql_query("SELECT * FROM TBL_Survey", conn)
    ade_records_df = pd.read_sql_query("SELECT * FROM TBL_ADERecords", conn)

    conn.close()
    return demographics_df, survey_df, ade_records_df

def merge_data(demographics_df, survey_df, ade_records_df):
    # Merge demographics and survey data
    data = pd.merge(demographics_df, survey_df, on='PatientID', how='left')

    # Create target variable
    patients_with_ade = ade_records_df['PatientID'].unique()
    data['Had_ADE'] = data['PatientID'].apply(lambda x: 1 if x in patients_with_ade else 0)

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
    data = data.drop(['PatientLast', 'PatientFirst', 'DOB', 'AdmissionDate', 'DischargeDate', 'AttributeName', 'SurveyDate'], axis=1)

    # Feature columns
    feature_columns = [col for col in data.columns if col != 'Had_ADE']

    # Ensure consistent feature ordering
    data = data[feature_columns + ['Had_ADE']]

    return data


def preprocess_individual_patient(patient_id):
    # Load the patient's data
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)

    # Get patient data
    demographics_query = f"SELECT * FROM TBL_Demographics WHERE PatientID = {patient_id}"
    demographics_df = pd.read_sql_query(demographics_query, conn)

    survey_query = f"SELECT * FROM TBL_Survey WHERE PatientID = {patient_id}"
    survey_df = pd.read_sql_query(survey_query, conn)

    conn.close()

    if demographics_df.empty or survey_df.empty:
        print(f"No data found for PatientID {patient_id}.")
        return None

    # Merge data
    data = pd.merge(demographics_df, survey_df, on='PatientID', how='left')

    # Preprocessing steps
    data = data.dropna()
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    data['AdmissionDate'] = pd.to_datetime(data['AdmissionDate'])
    data['DischargeDate'] = pd.to_datetime(data['DischargeDate'])
    data['LengthOfStay'] = (data['DischargeDate'] - data['AdmissionDate']).dt.days
    data = data.drop(['PatientLast', 'PatientFirst', 'DOB', 'AdmissionDate', 'DischargeDate', 'AttributeName', 'SurveyDate'], axis=1)

    # Feature columns
    feature_columns = ['Gender', 'Age', 'LengthOfStay', 'Score1', 'Score2', 'Score3', 'Score4']
    data = data[feature_columns]

    # Load scaler
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.joblib')
    if not os.path.exists(scaler_path):
        print("Scaler not found. Please ensure you've saved the scaler during model training.")
        return None
    scaler = joblib.load(scaler_path)

    # Scale features
    features_scaled = scaler.transform(data.values)

    return features_scaled
