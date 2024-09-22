import sqlite3
import os
import pandas as pd

def get_patient_demographics(patient_id):
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM TBL_Demographics WHERE PatientID = {patient_id}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_patient_surveys(patient_id):
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM TBL_Survey WHERE PatientID = {patient_id}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_patient_ade_records(patient_id):
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM TBL_ADERecords WHERE PatientID = {patient_id}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_patient_ade_records(patient_id):
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM TBL_ADERecords WHERE PatientID = {patient_id}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


