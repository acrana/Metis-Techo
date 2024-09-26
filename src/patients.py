# patients.py

patients = [
    {
        'PatientID': 1,
        'FirstName': 'John',
        'LastName': 'Doe',
        'Age': 58,
        'Gender': 'Male',
        'Allergies': ['Penicillin', 'Aspirin'],
        'MedicalHistory': ['Hypertension', 'Type 2 Diabetes', 'Atrial Fibrillation'],
        'LabResults': {
            'Blood Glucose': 145,          # mg/dL
            'Creatinine': 1.4,             # mg/dL
            'Potassium': 5.2,              # mmol/L (slightly elevated)
            'Sodium': 138,                 # mmol/L
            'TSH': 3.1,                    # uIU/mL (within normal)
            'INR': 2.8,                    # Therapeutic range (on Warfarin)
            'QTc': 480,                    # ms (prolonged QTc)
            'Troponin': 0.03,              # ng/mL (within normal limits)
            'Liver Enzymes': 35,           # U/L (within normal)
            'Hemoglobin A1C': 7.5,         # % (elevated, diabetic)
            'Blood Pressure': 150,         # mmHg (elevated)
        },
        'Medications': ['Warfarin', 'Metformin', 'Amiodarone', 'Lisinopril'],
        'CardiacMonitoring': {
            'ECG': 'Atrial Fibrillation',
            'QTc': 480,                    # Prolonged QTc interval
            'HeartRate': 75,               # bpm (controlled with medication)
        }
    },
    {
        'PatientID': 2,
        'FirstName': 'Jane',
        'LastName': 'Smith',
        'Age': 45,
        'Gender': 'Female',
        'Allergies': ['Ibuprofen', 'Sulfa drugs'],
        'MedicalHistory': ['Chronic Kidney Disease Stage 3', 'Hypertension'],
        'LabResults': {
            'Blood Glucose': 95,           # mg/dL (normal)
            'Creatinine': 2.1,             # mg/dL (elevated due to CKD)
            'Potassium': 4.5,              # mmol/L (normal)
            'Sodium': 136,                 # mmol/L (normal)
            'TSH': 4.2,                    # uIU/mL (borderline hypothyroid)
            'QTc': 420,                    # ms (normal)
            'eGFR': 38,                    # mL/min/1.73m2 (reduced kidney function)
            'UrineAlbumin': 250,           # mg/g (sign of kidney damage)
            'Blood Pressure': 135,         # mmHg (elevated)
        },
        'Medications': ['Losartan', 'Hydrochlorothiazide'],
        'CardiacMonitoring': {
            'ECG': 'Normal Sinus Rhythm',
            'QTc': 420,                    # ms (normal)
            'HeartRate': 78,               # bpm
        }
    },
    {
        'PatientID': 3,
        'FirstName': 'Michael',
        'LastName': 'Johnson',
        'Age': 68,
        'Gender': 'Male',
        'Allergies': [],
        'MedicalHistory': ['Coronary Artery Disease', 'Heart Failure with reduced EF', 'Chronic Obstructive Pulmonary Disease (COPD)'],
        'LabResults': {
            'Blood Glucose': 110,          # mg/dL (slightly elevated)
            'Creatinine': 1.2,             # mg/dL (normal)
            'Potassium': 4.9,              # mmol/L (high normal)
            'Sodium': 142,                 # mmol/L (normal)
            'NT-proBNP': 900,              # pg/mL (elevated, indicative of heart failure)
            'Troponin': 0.08,              # ng/mL (mild elevation, requires monitoring)
            'Liver Enzymes': 45,           # U/L (slightly elevated, possible liver congestion)
            'INR': 1.2,                    # Not on anticoagulants
            'QTc': 470,                    # ms (slightly prolonged)
            'Blood Pressure': 145,         # mmHg (elevated)
        },
        'Medications': ['Carvedilol', 'Spironolactone', 'Furosemide'],
        'CardiacMonitoring': {
            'ECG': 'Left Bundle Branch Block',
            'EjectionFraction': 35,        # % (reduced EF, heart failure)
            'QTc': 470,                    # ms (slightly prolonged)
            'HeartRate': 65,               # bpm (medicated control)
        }
    },
    {
        'PatientID': 4,
        'FirstName': 'Emily',
        'LastName': 'Williams',
        'Age': 52,
        'Gender': 'Female',
        'Allergies': ['ACE Inhibitors'],
        'MedicalHistory': ['Diabetes Mellitus Type 1', 'Peripheral Artery Disease', 'Hypertension'],
        'LabResults': {
            'Blood Glucose': 250,          # mg/dL (elevated)
            'Creatinine': 1.6,             # mg/dL (borderline kidney function)
            'Potassium': 5.3,              # mmol/L (elevated)
            'Sodium': 140,                 # mmol/L (normal)
            'Hemoglobin A1C': 9.2,         # % (poor glucose control)
            'TSH': 2.8,                    # uIU/mL (normal)
            'QTc': 500,                    # ms (significantly prolonged)
            'Blood Pressure': 160,         # mmHg (severely elevated)
            'GFR': 55,                     # mL/min/1.73m2 (mildly reduced)
        },
        'Medications': ['Insulin', 'Amlodipine'],
        'CardiacMonitoring': {
            'ECG': 'Ventricular Tachycardia',
            'QTc': 500,                    # ms (prolonged, risk of arrhythmias)
            'HeartRate': 100,              # bpm (tachycardic)
        }
    },
    {
        'PatientID': 5,
        'FirstName': 'David',
        'LastName': 'Martinez',
        'Age': 62,
        'Gender': 'Male',
        'Allergies': ['Statins'],
        'MedicalHistory': ['Chronic Liver Disease', 'Hypertension', 'Atrial Fibrillation', 'Hyperkalemia'],
        'LabResults': {
            'Blood Glucose': 98,           # mg/dL (normal)
            'Creatinine': 1.3,             # mg/dL (slightly elevated)
            'Potassium': 6.1,              # mmol/L (dangerously elevated)
            'Sodium': 137,                 # mmol/L (normal)
            'Liver Enzymes': 95,           # U/L (elevated, liver disease)
            'Bilirubin': 2.2,              # mg/dL (elevated, liver disease)
            'INR': 2.1,                    # Elevated due to liver dysfunction
            'QTc': 490,                    # ms (prolonged)
            'Blood Pressure': 140,         # mmHg (elevated)
        },
        'Medications': ['Spironolactone', 'Digoxin'],
        'CardiacMonitoring': {
            'ECG': 'Atrial Fibrillation',
            'QTc': 490,                    # ms (prolonged)
            'HeartRate': 90,               # bpm
        }
    }
]
