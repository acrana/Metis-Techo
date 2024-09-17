
import sqlite3

def setup_database():
    conn = sqlite3.connect('data/clinical_decision_support.db')
    cursor = conn.cursor()

    # Create TBL_Demographics
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS TBL_Demographics (
        PatientID INTEGER PRIMARY KEY,
        PatientLast TEXT NOT NULL,
        PatientFirst TEXT NOT NULL,
        Gender TEXT,
        DOB DATE,
        Age INTEGER,
        AdmissionDate DATE,
        DischargeDate DATE
    )
    ''')

    # Create TBL_Survey
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS TBL_Survey (
        AttributeName TEXT PRIMARY KEY,
        PatientID INTEGER,
        SurveyDate DATE,
        TrustInMedicationScore INTEGER,
        UnderstandingMedicationScore INTEGER,
        SafetyScore INTEGER,
        OverallSatisfactionScore INTEGER,
        FOREIGN KEY (PatientID) REFERENCES TBL_Demographics(PatientID)
    )
    ''')

    # Create TBL_ADERecords
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS TBL_ADERecords (
        ADE_RecordID INTEGER PRIMARY KEY,
        PatientID INTEGER,
        DateOfIncident DATE,
        MedicationInvolved TEXT,
        Severity TEXT,
        TreatmentGiven TEXT,
        FOREIGN KEY (PatientID) REFERENCES TBL_Demographics(PatientID)
    )
    ''')

    # Insert sample data into TBL_Demographics
    demographics_data = [
        (1, 'Doe', 'John', 'Male', '1980-05-15', 43, '2023-10-01', '2023-10-15'),
        (2, 'Smith', 'Jane', 'Female', '1990-08-22', 33, '2023-10-05', '2023-10-20')
    ]
    cursor.executemany('''
    INSERT OR IGNORE INTO TBL_Demographics (PatientID, PatientLast, PatientFirst, Gender, DOB, Age, AdmissionDate, DischargeDate)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', demographics_data)

    # Insert sample data into TBL_Survey
    survey_data = [
        ('Survey1', 1, '2023-10-10', 8, 7, 9, 8),
        ('Survey2', 2, '2023-10-12', 9, 8, 8, 9)
    ]
    cursor.executemany('''
    INSERT OR IGNORE INTO TBL_Survey (AttributeName, PatientID, SurveyDate, TrustInMedicationScore, UnderstandingMedicationScore, SafetyScore, OverallSatisfactionScore)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', survey_data)

    # Insert sample data into TBL_ADERecords
    ade_records_data = [
        (1, 1, '2023-10-11', 'Medication A', 'Moderate', 'Antidote A'),
        (2, 2, '2023-10-13', 'Medication B', 'Severe', 'Antidote B')
    ]
    cursor.executemany('''
    INSERT OR IGNORE INTO TBL_ADERecords (ADE_RecordID, PatientID, DateOfIncident, MedicationInvolved, Severity, TreatmentGiven)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', ade_records_data)

    conn.commit()
    conn.close()
    print("Database setup completed successfully.")

if __name__ == '__main__':
    setup_database()
