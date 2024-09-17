
import sqlite3

DB_PATH = 'data/clinical_decision_support.db'

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

def get_patient_demographics(patient_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT * FROM TBL_Demographics WHERE PatientID = ?
    ''', (patient_id,))
    result = cursor.fetchone()
    conn.close()
    return result

def get_patient_surveys(patient_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT * FROM TBL_Survey WHERE PatientID = ?
    ''', (patient_id,))
    result = cursor.fetchall()
    conn.close()
    return result

def get_patient_ade_records(patient_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT * FROM TBL_ADERecords WHERE PatientID = ?
    ''', (patient_id,))
    result = cursor.fetchall()
    conn.close()
    return result
