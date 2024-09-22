# src/preprocessing.py

import sqlite3
import pandas as pd

# Function to load data from SQLite database
def load_data(db_path='data/clinical_decision_support.db'):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Read tables into pandas DataFrames
    demographics_df = pd.read_sql_query("SELECT * FROM TBL_Demographics", conn)
    survey_df = pd.read_sql_query("SELECT * FROM TBL_Survey", conn)
    ade_records_df = pd.read_sql_query("SELECT * FROM TBL_ADERecords", conn)
    
    # Close the connection
    conn.close()
    
    return demographics_df, survey_df, ade_records_df

# Function to merge data and create target variable
def merge_data(demographics_df, survey_df, ade_records_df):
    # Merge demographics and survey data on PatientID
    data = pd.merge(demographics_df, survey_df, on='PatientID', how='left')
    
    # Create a list of patients who had an ADE
    patients_with_ade = ade_records_df['PatientID'].unique()
    
    # Add a binary target column to the data
    data['Had_ADE'] = data['PatientID'].apply(lambda x: 1 if x in patients_with_ade else 0)
    
    return data

# Function to clean data and perform feature engineering
def preprocess_data(data):
    # Handle missing values (if any)
    data = data.dropna()
    
    # Convert categorical variables to numeric encoding
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    
    # Convert dates to datetime objects and extract useful features
    data['AdmissionDate'] = pd.to_datetime(data['AdmissionDate'])
    data['DischargeDate'] = pd.to_datetime(data['DischargeDate'])
    data['LengthOfStay'] = (data['DischargeDate'] - data['AdmissionDate']).dt.days
    
    # Drop irrelevant columns
    data = data.drop(['PatientLast', 'PatientFirst', 'DOB', 'AdmissionDate', 'DischargeDate', 'AttributeName', 'SurveyDate'], axis=1)
    
    return data

# Main function to load and preprocess data
def get_preprocessed_data():
    demographics_df, survey_df, ade_records_df = load_data()
    data = merge_data(demographics_df, survey_df, ade_records_df)
    data = preprocess_data(data)
    return data

# Add this function to src/preprocessing.py

def preprocess_individual_patient(patient_id):
    import sqlite3
    import pandas as pd
    import numpy as np
    import os
    import joblib

    # Load the patient's data from the database
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)

    # Get patient demographics
    demographics_query = f"SELECT * FROM TBL_Demographics WHERE PatientID = {patient_id}"
    demographics_df = pd.read_sql_query(demographics_query, conn)

    # Get patient survey data
    survey_query = f"SELECT * FROM TBL_Survey WHERE PatientID = {patient_id}"
    survey_df = pd.read_sql_query(survey_query, conn)

    conn.close()

    # Check if patient data exists
    if demographics_df.empty or survey_df.empty:
        print(f"No data found for PatientID {patient_id}.")
        return None

    # Merge demographics and survey data
    data = pd.merge(demographics_df, survey_df, on='PatientID', how='left')

    # Preprocess the data similar to the preprocess_data function
    # Handle missing values
    data = data.dropna()

    # Convert categorical variables to numeric encoding
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    # Convert dates to datetime objects and extract useful features
    data['AdmissionDate'] = pd.to_datetime(data['AdmissionDate'])
    data['DischargeDate'] = pd.to_datetime(data['DischargeDate'])
    data['LengthOfStay'] = (data['DischargeDate'] - data['AdmissionDate']).dt.days

    # Drop irrelevant columns
    data = data.drop(['PatientLast', 'PatientFirst', 'DOB', 'AdmissionDate', 'DischargeDate', 'AttributeName', 'SurveyDate'], axis=1)

    # Drop 'PatientID' and 'Had_ADE' if present
    data = data.drop(['PatientID', 'Had_ADE'], axis=1, errors='ignore')

    # Ensure the columns are in the same order as during training
    feature_columns = ['Gender', 'Age', 'LengthOfStay', 'Score1', 'Score2', 'Score3', 'Score4']
    data = data[feature_columns]

    # Convert DataFrame to numpy array
    features = data.values

    # Load the scaler used during training
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.joblib')
    if not os.path.exists(scaler_path):
        print("Scaler not found. Please ensure you've saved the scaler during model training.")
        return None
    scaler = joblib.load(scaler_path)

    # Scale the features
    features_scaled = scaler.transform(features)

    return features_scaled


# If this script is run directly, demonstrate data loading and preprocessing
if __name__ == '__main__':
    data = get_preprocessed_data()
    print(data.head())
