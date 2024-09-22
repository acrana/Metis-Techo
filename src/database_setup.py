# src/database_setup.py

import sqlite3
import os

def setup_database():
    # Database path
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop tables if they exist (to start fresh)
    cursor.execute("DROP TABLE IF EXISTS TBL_Demographics")
    cursor.execute("DROP TABLE IF EXISTS TBL_Survey")
    cursor.execute("DROP TABLE IF EXISTS TBL_ADERecords")
    cursor.execute("DROP TABLE IF EXISTS TBL_Medications")
    cursor.execute("DROP TABLE IF EXISTS TBL_Prescriptions")

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS TBL_Demographics (
            PatientID INTEGER PRIMARY KEY,
            PatientLast TEXT,
            PatientFirst TEXT,
            Gender TEXT,
            DOB TEXT,
            AdmissionDate TEXT,
            DischargeDate TEXT,
            Age INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS TBL_Survey (
            SurveyID INTEGER PRIMARY KEY AUTOINCREMENT,
            PatientID INTEGER,
            AttributeName TEXT,
            Score1 INTEGER,
            Score2 INTEGER,
            Score3 INTEGER,
            Score4 INTEGER,
            SurveyDate TEXT,
            FOREIGN KEY (PatientID) REFERENCES TBL_Demographics (PatientID)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS TBL_ADERecords (
            ADEID INTEGER PRIMARY KEY AUTOINCREMENT,
            PatientID INTEGER,
            Medication TEXT,
            ADEDescription TEXT,
            Date TEXT,
            FOREIGN KEY (PatientID) REFERENCES TBL_Demographics (PatientID)
        )
    ''')

    # Insert sample patient demographics
    cursor.executemany('''
        INSERT INTO TBL_Demographics (PatientID, PatientLast, PatientFirst, Gender, DOB, AdmissionDate, DischargeDate, Age)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', [
        (1, 'Doe', 'John', 'Male', '1978-04-12', '2023-09-01', '2023-09-10', 45),
        (2, 'Smith', 'Jane', 'Female', '1963-11-23', '2023-09-05', '2023-09-15', 60),
        (3, 'Johnson', 'Alice', 'Female', '1988-07-19', '2023-09-08', '2023-09-18', 35),
        (4, 'Brown', 'Robert', 'Male', '1973-02-05', '2023-09-12', '2023-09-22', 50),
        (5, 'Davis', 'Emily', 'Female', '1995-05-30', '2023-09-15', '2023-09-25', 28),
    ])

    # Insert sample surveys
    cursor.executemany('''
        INSERT INTO TBL_Survey (PatientID, AttributeName, Score1, Score2, Score3, Score4, SurveyDate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', [
        (1, 'Well-being Survey', 7, 8, 7, 8, '2023-09-02'),
        (2, 'Diabetes Management Survey', 6, 5, 7, 6, '2023-09-06'),
        (3, 'Post-Surgery Survey', 8, 9, 8, 7, '2023-09-10'),
        (4, 'Cardiac Health Survey', 5, 6, 5, 6, '2023-09-14'),
        (5, 'Infection Symptoms Survey', 7, 7, 8, 8, '2023-09-16'),
    ])

    # Insert sample ADE records
    cursor.executemany('''
        INSERT INTO TBL_ADERecords (PatientID, Medication, ADEDescription, Date)
        VALUES (?, ?, ?, ?)
    ''', [
        (2, 'Insulin', 'Experienced hypoglycemia due to insulin overdose', '2023-09-07'),
        (3, 'Penicillin', 'Developed an allergic reaction: hives and swelling', '2023-09-11'),
        (5, 'Ibuprofen', 'Gastrointestinal bleeding after NSAID use', '2023-09-18'),
    ])

    # Create Medication Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS TBL_Medications (
            MedicationID INTEGER PRIMARY KEY AUTOINCREMENT,
            MedicationName TEXT UNIQUE,
           



