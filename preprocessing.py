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

# If this script is run directly, demonstrate data loading and preprocessing
if __name__ == '__main__':
    data = get_preprocessed_data()
    print(data.head())
