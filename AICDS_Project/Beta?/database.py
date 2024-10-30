# database.py
import sqlite3
from sqlalchemy import create_engine
import pandas as pd

def create_tables(engine):
    """
    Creates all necessary tables in the SQLite database.
    """
    with engine.connect() as conn:
        # Create Patients table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Patients (
                patient_id INTEGER PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                date_of_birth TEXT,
                gender TEXT,
                phone TEXT,
                email TEXT,
                address TEXT
            )
        """)
        
        # Create Providers table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Providers (
                provider_id INTEGER PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                role TEXT,
                phone TEXT,
                email TEXT
            )
        """)
        
        # Create Medications table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Medications (
                medication_id INTEGER PRIMARY KEY,
                name TEXT,
                description TEXT
            )
        """)
        
        # Create Orders table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Orders (
                order_id INTEGER PRIMARY KEY,
                patient_id INTEGER,
                provider_id INTEGER,
                medication_id INTEGER,
                order_date TEXT,
                dosage TEXT,
                frequency TEXT,
                duration TEXT,
                status TEXT,
                FOREIGN KEY(patient_id) REFERENCES Patients(patient_id),
                FOREIGN KEY(provider_id) REFERENCES Providers(provider_id),
                FOREIGN KEY(medication_id) REFERENCES Medications(medication_id)
            )
        """)
        
        # Create Lab_Results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Lab_Results (
                lab_result_id INTEGER PRIMARY KEY,
                patient_id INTEGER,
                test_name TEXT,
                result_value REAL,
                units TEXT,
                test_date TEXT,
                FOREIGN KEY(patient_id) REFERENCES Patients(patient_id)
            )
        """)
        
        # Create Medication_Lab_Effects table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Medication_Lab_Effects (
                effect_id INTEGER PRIMARY KEY,
                medication_id INTEGER,
                lab_test_name TEXT,
                impact TEXT,
                threshold REAL,
                alert_message TEXT,
                FOREIGN KEY(medication_id) REFERENCES Medications(medication_id)
            )
        """)
        
        # Create Alerts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Alerts (
                alert_id INTEGER PRIMARY KEY,
                patient_id INTEGER,
                medication_id INTEGER,
                lab_test_name TEXT,
                alert_message TEXT,
                alert_date TEXT,
                resolved INTEGER,
                FOREIGN KEY(patient_id) REFERENCES Patients(patient_id),
                FOREIGN KEY(medication_id) REFERENCES Medications(medication_id)
            )
        """)
        
        # Create Alert_History table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Alert_History (
                history_id INTEGER PRIMARY KEY,
                alert_id INTEGER,
                action_taken TEXT,
                action_date TEXT,
                FOREIGN KEY(alert_id) REFERENCES Alerts(alert_id)
            )
        """)

def insert_sample_data(engine):
    """
    Inserts sample data into the tables.
    """
    with engine.connect() as conn:
        # Sample Patients
        patients = pd.DataFrame([
            [1, 'John', 'Doe', '1980-05-15', 'Male', '123-456-7890', 'john.doe@example.com', '123 Elm Street'],
            [2, 'Jane', 'Smith', '1975-08-22', 'Female', '987-654-3210', 'jane.smith@example.com', '456 Oak Avenue'],
            # Add more patients as needed
            [3, 'Alice', 'Johnson', '1990-12-10', 'Female', '555-123-4567', 'alice.johnson@example.com', '789 Pine Road']
        ], columns=['patient_id', 'first_name', 'last_name', 'date_of_birth', 'gender', 'phone', 'email', 'address'])
        patients.to_sql('Patients', conn, if_exists='append', index=False)
        
        # Sample Providers
        providers = pd.DataFrame([
            [1, 'Emily', 'Clark', 'Cardiologist', '456-789-0123', 'emily.clark@hospital.com'],
            [2, 'Michael', 'Brown', 'Neurologist', '567-890-1234', 'michael.brown@hospital.com'],
            # Add more providers as needed
            [3, 'Sarah', 'Williams', 'Oncologist', '678-901-2345', 'sarah.williams@hospital.com']
        ], columns=['provider_id', 'first_name', 'last_name', 'role', 'phone', 'email'])
        providers.to_sql('Providers', conn, if_exists='append', index=False)
        
        # Sample Medications
        medications = pd.DataFrame([
            [1, 'Amiodarone', 'Used to treat and prevent certain types of irregular heartbeat.'],
            [2, 'Atorvastatin', 'Used to lower cholesterol and triglycerides in the blood.'],
            # Add more medications as needed
            [3, 'Lisinopril', 'Used to treat high blood pressure and heart failure.']
        ], columns=['medication_id', 'name', 'description'])
        medications.to_sql('Medications', conn, if_exists='append', index=False)
        
        # Sample Orders
        orders = pd.DataFrame([
            [1, 1, 1, 1, '2024-04-15 14:00:00', '200 mg', 'Once daily', '30 days', 'Active'],
            [2, 2, 2, 2, '2024-04-16 10:30:00', '10 mg', 'Twice daily', '60 days', 'Active'],
            # Add more orders as needed
            [3, 3, 3, 3, '2024-04-17 09:15:00', '5 mg', 'Once daily', '90 days', 'Active'],
            # Adding orders that should trigger alerts
            [4, 1, 1, 1, '2024-04-18 11:00:00', '300 mg', 'Once daily', '30 days', 'Active']
        ], columns=['order_id', 'patient_id', 'provider_id', 'medication_id', 'order_date', 'dosage', 'frequency', 'duration', 'status'])
        orders.to_sql('Orders', conn, if_exists='append', index=False)
        
        # Sample Lab Results
        lab_results = pd.DataFrame([
            [1, 1, 'QTC', 450, 'ms', '2024-04-10'],
            [2, 2, 'LDL', 130, 'mg/dL', '2024-04-11'],
            # Add more lab results as needed
            [3, 3, 'Blood Pressure', 140, 'mmHg', '2024-04-12'],
            [4, 1, 'QTC', 460, 'ms', '2024-04-16']
        ], columns=['lab_result_id', 'patient_id', 'test_name', 'result_value', 'units', 'test_date'])
        lab_results.to_sql('Lab_Results', conn, if_exists='append', index=False)
        
        # Sample Medication_Lab_Effects
        med_lab_effects = pd.DataFrame([
            [1, 1, 'QTC', 'Increase', 440, 'High QTC interval detected with Amiodarone. Risk of arrhythmia.'],
            [2, 2, 'LDL', 'Decrease', 100, 'LDL cholesterol is below target.'],
            # Add more effects as needed
            [3, 3, 'Blood Pressure', 'Decrease', 90, 'Blood pressure is below normal range. Monitor closely.']
        ], columns=['effect_id', 'medication_id', 'lab_test_name', 'impact', 'threshold', 'alert_message'])
        med_lab_effects.to_sql('Medication_Lab_Effects', conn, if_exists='append', index=False)
        
        # Sample Alerts (initially empty)
        alerts = pd.DataFrame(columns=['alert_id', 'patient_id', 'medication_id', 'lab_test_name', 'alert_message', 'alert_date', 'resolved'])
        alerts.to_sql('Alerts', conn, if_exists='append', index=False)
        
        # Sample Alert_History (initially empty)
        alert_history = pd.DataFrame(columns=['history_id', 'alert_id', 'action_taken', 'action_date'])
        alert_history.to_sql('Alert_History', conn, if_exists='append', index=False)

def reset_database(engine):
    """
    Drops all tables and recreates them. Use with caution.
    """
    with engine.connect() as conn:
        conn.execute("DROP TABLE IF EXISTS Alert_History")
        conn.execute("DROP TABLE IF EXISTS Alerts")
        conn.execute("DROP TABLE IF EXISTS Medication_Lab_Effects")
        conn.execute("DROP TABLE IF EXISTS Lab_Results")
        conn.execute("DROP TABLE IF EXISTS Orders")
        conn.execute("DROP TABLE IF EXISTS Medications")
        conn.execute("DROP TABLE IF EXISTS Providers")
        conn.execute("DROP TABLE IF EXISTS Patients")
        print("All tables dropped successfully.")
        create_tables(engine)
        insert_sample_data(engine)
        print("Database reset and sample data inserted successfully.")

if __name__ == "__main__":
    engine = create_engine('sqlite:///clinical_data.db')
    create_tables(engine)
    insert_sample_data(engine)
    print("Database and sample data setup completed successfully.")


