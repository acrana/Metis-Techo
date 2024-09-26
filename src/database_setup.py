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

    # Create tables (include your CREATE TABLE statements here)
    # ...

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

    # ... [Include other table creation code here]

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

    # [Include the rest of your data generation code]

    # Commit changes and close connection
    conn.commit()
    conn.close()
    print("Database setup complete with synthetic data.")

if __name__ == '__main__':
    setup_database()

