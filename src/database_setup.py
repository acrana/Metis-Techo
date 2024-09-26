# src/database_setup.py

import sqlite3
import os
import random
from faker import Faker
import pandas as pd

def setup_database(num_patients=1000, num_medications=100, num_prescriptions=5000, num_ade_records=500, num_surveys=2000):
    fake = Faker()
    Faker.seed(0)
    random.seed(0)

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

    # Create Demographics Table
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

    # Create Survey Table
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

    # Create ADERecords Table
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

    # Create Medications Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS TBL_Medications (
            MedicationID INTEGER PRIMARY KEY AUTOINCREMENT,
            MedicationName TEXT UNIQUE,
            RiskFactors TEXT  -- This can be a JSON string or delimited list
        )
    ''')

    # Create Prescriptions Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS TBL_Prescriptions (
            PrescriptionID INTEGER PRIMARY KEY AUTOINCREMENT,
            PatientID INTEGER,
            MedicationID INTEGER,
            Date TEXT,
            FOREIGN KEY (PatientID) REFERENCES TBL_Demographics (PatientID),
            FOREIGN KEY (MedicationID) REFERENCES TBL_Medications (MedicationID)
        )
    ''')

    # Create Medications
    medications = []
    risk_factors_list = ["Pregnancy", "Renal impairment", "Metabolic acidosis", "Liver disease", "Angioedema", "Penicillin allergy", "Gastrointestinal bleeding"]
    for i in range(1, num_medications + 1):
        medication_name = fake.unique.word().capitalize()
        risk_factors = ', '.join(random.sample(risk_factors_list, k=random.randint(1, 3)))
        medications.append((i, medication_name, risk_factors))

    cursor.executemany('''
        INSERT INTO TBL_Medications (MedicationID, MedicationName, RiskFactors)
        VALUES (?, ?, ?)
    ''', medications)

    # Create Patients
    patients = []
    genders = ["Male", "Female"]
    for i in range(1, num_patients + 1):
        first_name = fake.first_name()
        last_name = fake.last_name()
        gender = random.choice(genders)
        dob = fake.date_of_birth(minimum_age=20, maximum_age=90).strftime("%Y-%m-%d")

        # Keep admission_date as a date object
        admission_date = fake.date_between(start_date='-2y', end_date='today')

        # Generate discharge_date using the admission_date date object
        discharge_date = fake.date_between(start_date=admission_date, end_date='today')

        # Now format dates to strings for database insertion
        admission_date_str = admission_date.strftime("%Y-%m-%d")
        discharge_date_str = discharge_date.strftime("%Y-%m-%d")

        age = 2024 - int(dob[:4])

        patients.append((i, last_name, first_name, gender, dob, admission_date_str, discharge_date_str, age))

    cursor.executemany('''
        INSERT INTO TBL_Demographics (PatientID, PatientLast, PatientFirst, Gender, DOB, AdmissionDate, DischargeDate, Age)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', patients)

    # Create Prescriptions
    prescriptions = []
    for i in range(1, num_prescriptions + 1):
        patient_id = random.randint(1, num_patients)
        medication_id = random.randint(1, num_medications)
        date = fake.date_between(start_date='-2y', end_date='today').strftime("%Y-%m-%d")
        prescriptions.append((patient_id, medication_id, date))

    cursor.executemany('''
        INSERT INTO TBL_Prescriptions (PatientID, MedicationID, Date)
        VALUES (?, ?, ?)
    ''', prescriptions)

    # Create ADE Records
    ade_descriptions = ["Allergic reaction", "Nausea", "Vomiting", "Dizziness", "Headache", "Rash", "Hypoglycemia"]
    ade_records = []
    for i in range(1, num_ade_records + 1):
        patient_id = random.randint(1, num_patients)
        medication_id = random.randint(1, num_medications)
        medication_name = medications[medication_id - 1][1]  # Get medication name from medications list
        ade_description = random.choice(ade_descriptions)
        date = fake.date_between(start_date='-2y', end_date='today').strftime("%Y-%m-%d")
        ade_records.append((patient_id, medication_name, ade_description, date))

    cursor.executemany('''
        INSERT INTO TBL_ADERecords (PatientID, Medication, ADEDescription, Date)
        VALUES (?, ?, ?, ?)
    ''', ade_records)

    # Create Surveys
    surveys = []
    survey_types = ["Well-being Survey", "Diabetes Management Survey", "Post-Surgery Survey", "Cardiac Health Survey", "Infection Symptoms Survey"]
    for i in range(1, num_surveys + 1):
        patient_id = random.randint(1, num_patients)
        attribute_name = random.choice(survey_types)
        scores = [random.randint(1, 10) for _ in range(4)]
        survey_date = fake.date_between(start_date='-2y', end_date='today').strftime("%Y-%m-%d")
        surveys.append((patient_id, attribute_name, *scores, survey_date))

    cursor.executemany('''
        INSERT INTO TBL_Survey (PatientID, AttributeName, Score1, Score2, Score3, Score4, SurveyDate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', surveys)

    # Commit changes and close connection
    conn.commit()
    conn.close()
    print("Database setup complete with synthetic data.")

if __name__ == '__main__':
    setup_database()

