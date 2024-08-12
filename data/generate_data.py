# src/generate_data.py
import pandas as pd
import numpy as np

# Generate simple patient data
patient_data = {
    'PatientID': range(1, 101),
    'Age': np.random.randint(18, 90, 100),
    'Gender': np.random.choice(['Male', 'Female'], 100),
    'BloodPressure': np.random.randint(90, 160, 100),
    'Cholesterol': np.random.randint(100, 300, 100),
    'Diabetes': np.random.choice([0, 1], 100)
}

# Convert to DataFrame
df_patients = pd.DataFrame(patient_data)

# Save to CSV
df_patients.to_csv('data/simpledata.csv', index=False)

# Generate simple medication data
medication_data = {
    'MedicationID': range(1, 101),
    'PatientID': range(1, 101),
    'Medication': np.random.choice(['MedA', 'MedB', 'MedC'], 100),
    'Dosage': np.random.randint(1, 100, 100),
    'Outcome': np.random.choice([0, 1], 100)
}

# Convert to DataFrame
df_medications = pd.DataFrame(medication_data)

# Save to CSV
df_medications.to_csv('data/simplemedications.csv', index=False)

print("Data generated and saved.")
