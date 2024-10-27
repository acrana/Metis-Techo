# src/data_access.py

import sqlite3
import os
import pandas as pd

def get_patient_demographics(patient_id):
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM TBL_Demographics WHERE PatientID = ?"
    df = pd.read_sql_query(query, conn, params=(patient_id,))
    conn.close()
    return df

def get_patient_surveys(patient_id):
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM TBL_Survey WHERE PatientID = ?"
    df = pd.read_sql_query(query, conn, params=(patient_id,))
    conn.close()
    return df

def get_patient_ade_records(patient_id):
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM TBL_ADERecords WHERE PatientID = ?"
    df = pd.read_sql_query(query, conn, params=(patient_id,))
    conn.close()
    return df

def get_medication_info(medication_name):
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM TBL_Medications WHERE MedicationName = ?"
    df = pd.read_sql_query(query, conn, params=(medication_name,))
    conn.close()
    return df

def add_prescription(patient_id, medication_name):
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get MedicationID
    cursor.execute("SELECT MedicationID FROM TBL_Medications WHERE MedicationName = ?", (medication_name,))
    result = cursor.fetchone()
    if result:
        medication_id = result[0]
    else:
        print(f"Medication '{medication_name}' not found in database.")
        conn.close()
        return False

    # Insert Prescription
    cursor.execute('''
        INSERT INTO TBL_Prescriptions (PatientID, MedicationID, Date)
        VALUES (?, ?, date('now'))
    ''', (patient_id, medication_id))

    conn.commit()
    conn.close()
    print(f"Prescription for '{medication_name}' added for Patient ID {patient_id}.")
    return True


