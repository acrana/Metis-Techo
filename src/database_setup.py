import sqlite3
import os

def setup_database():
    # Database path
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_decision_support.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

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
            ADEDescription TEXT,
            Date TEXT,
            FOREIGN KEY (PatientID) REFERENCES TBL_Demographics (PatientID)
        )
    ''')

    # Insert sample data
    cursor.execute('''
        INSERT INTO TBL_Demographics (PatientID, PatientLast, PatientFirst, Gender, DOB, AdmissionDate, DischargeDate, Age)
        VALUES
            (1, 'Doe', 'John', 'Male', '1980-05-15', '2023-09-01', '2023-09-10', 43),
            (2, 'Smith', 'Jane', 'Female', '1975-08-20', '2023-09-05', '2023-09-15', 48)
    ''')

    cursor.execute('''
        INSERT INTO TBL_Survey (PatientID, AttributeName, Score1, Score2, Score3, Score4, SurveyDate)
        VALUES
            (1, 'Initial Survey', 8, 7, 9, 8, '2023-09-02'),
            (2, 'Initial Survey', 6, 5, 7, 6, '2023-09-06')
    ''')

    cursor.execute('''
        INSERT INTO TBL_ADERecords (PatientID, ADEDescription, Date)
        VALUES
            (2, 'Mild rash after medication', '2023-09-08')
    ''')

    # Commit changes and close connection
    conn.commit()
    conn.close()
    print("Database setup complete.")

if __name__ == '__main__':
    setup_database()

